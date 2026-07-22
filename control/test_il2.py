import csv
import unittest
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

import numpy as np

import control.il2 as il2
from control.teleop_input import GamepadSample
from control.teleop_runtime import FeedRuntimeState, TeleopRuntimeContext


def build_yaml_text(
    *,
    J_rows: str,
    motor_limit_mm: float = 10.0,
    yaw_limit_deg: float = 170.0,
    pitch_limit_deg: float = 170.0,
    motor_order: str = '["m1", "m2", "m3", "m4"]',
    ids: str = "[1, 2, 3, 4]",
    id_by_motor: str = "m1: 1\n      m2: 2\n      m3: 3\n      m4: 4",
) -> str:
    return f"""
sim2real:
  J_4x2_mm_per_rad:
{J_rows}
  yaw_limit_deg: {yaw_limit_deg}
  pitch_limit_deg: {pitch_limit_deg}
  motor_limit_mm: {motor_limit_mm}
  control_hz: 30.0
  motor_order: {motor_order}
  serial:
    port: "/dev/ttyUSB0"
    baudrate: 115200
    timeout: 0.0
    write_timeout: 0.2
    critical_retry_count: 3
    critical_retry_interval_sec: 0.1
  actuator:
    mode: broadcast_follow_no_feedback
    ids: {ids}
    id_by_motor:
      {id_by_motor}
    position_index: 0x37
    count_min: 0
    count_max: 2000
    per_motor:
      m1:
        zero_count: 946
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 500
        soft_max_count: 1500
      m2:
        zero_count: 1095
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 500
        soft_max_count: 1500
      m3:
        zero_count: 1054
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 500
        soft_max_count: 1500
      m4:
        zero_count: 905
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 500
        soft_max_count: 1500
  alarm:
    enabled: true
    repeat: 3
    terminal_bell: true
    banner_width: 70
  monitor:
    enabled: false
    query_hz: 5.0
    response_timeout_sec: 0.1
    failure_threshold: 15
    error_mask: 0x0F
    log_every_n: 8
  teleop:
    controller_index: 0
    invert_yaw: false
    invert_pitch: true
    disconnect_timeout_sec: 0.5
    allow_feed_without_actuator: false
    left_stick_x_axis: 0
    left_stick_y_axis: 1
  feed:
    teleop_enabled: true
    port: "/dev/ttyUSB1"
    baudrate: 115200
    timeout: 0.05
    addr: 1
    microstep: 16
    steps_per_rev: 200
    step_pulses: 60
    backward_step_pulses: 100
    repeat_hz: 30
    default_vel: 100
    default_acc: 0
    enable_on_start: true
  output:
    print_tx_frame: false
    print_every_n: 0
    save_csv: false
    csv_path: "./sim2real_motor_log.csv"
    save_plot: false
    plot_path: "./sim2real_motor_plot.png"
    plot_dpi: 120
    max_plot_points: 4000
"""


class IL2Tests(unittest.TestCase):
    def _write_yaml(self, text: str) -> str:
        fp = NamedTemporaryFile("w+", suffix=".yaml", encoding="utf-8", delete=False)
        self.addCleanup(lambda: __import__("os").unlink(fp.name))
        fp.write(text)
        fp.flush()
        fp.close()
        return fp.name

    def test_same_policy_output_uses_yaml_matrix(self) -> None:
        yaml_a = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]"
            )
        )
        yaml_b = self._write_yaml(
            build_yaml_text(
                J_rows="    - [0.0, 5.0]\n    - [5.0, 0.0]\n    - [0.0, -5.0]\n    - [-5.0, 0.0]"
            )
        )

        control_a = il2.load_yaml_bound_control(yaml_a)
        control_b = il2.load_yaml_bound_control(yaml_b)

        result_a = il2.apply_policy_action(control_a, action=(0.1, 0.2), dt=1.0 / 30.0)
        result_b = il2.apply_policy_action(control_b, action=(0.1, 0.2), dt=1.0 / 30.0)

        self.assertFalse(np.allclose(result_a.motor_target_mm, result_b.motor_target_mm))

    def test_motor_order_changes_actuator_id_order(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]",
                motor_order='["m2", "m1", "m4", "m3"]',
                ids="[4, 3, 2, 1]",
                id_by_motor="m1: 4\n      m2: 3\n      m3: 2\n      m4: 1",
            )
        )

        control = il2.load_yaml_bound_control(yaml_path)

        self.assertEqual(control.actuator_ids, (3, 4, 1, 2))

    def test_policy_angle_limit_returns_estop(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]",
                yaw_limit_deg=1.0,
            )
        )
        control = il2.load_yaml_bound_control(yaml_path)

        result = il2.apply_policy_action(control, action=(np.deg2rad(2.0), 0.0), dt=1.0 / 30.0)

        self.assertEqual(result.command, il2.BridgeCommand.ESTOP)
        self.assertTrue(control.sim2real_bridge.estop_latched)

    def test_motor_target_candidate_is_clamped_by_motor_limit(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [300.0, 0.0]\n    - [-300.0, 0.0]\n    - [0.0, 300.0]\n    - [0.0, -300.0]",
                motor_limit_mm=10.0,
            )
        )
        control = il2.load_yaml_bound_control(yaml_path)

        result = il2.apply_policy_action(control, action=(0.1, 0.1), dt=1.0 / 30.0)

        np.testing.assert_allclose(
            result.motor_target_mm,
            np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float64),
        )

    def test_mm_target_to_counts_is_clamped_to_soft_and_hard_limits(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]"
            )
        )
        control = il2.load_yaml_bound_control(yaml_path)

        counts = il2.map_motor_target_to_counts(
            control,
            motor_target_mm=np.array([100.0, 100.0, -100.0, -100.0], dtype=np.float64),
        )

        np.testing.assert_array_equal(counts, np.array([1500, 1500, 500, 500], dtype=np.int32))

    def test_feed_buttons_do_not_change_yaw_pitch_motor_target(self) -> None:
        feed_state = FeedRuntimeState(
            enabled=True,
            cfg=type("Cfg", (), {"repeat_hz": 30.0})(),
            serial_link=None,
            current_pulses=0,
            locked=False,
            next_send_ts=0.0,
        )
        ctx = TeleopRuntimeContext(
            sim2real_bridge=type(
                "Bridge",
                (),
                {
                    "estop_latched": False,
                    "motor_target_mm": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                },
            )(),
            motor_mapper=None,
            actuator_ids=(),
            actuator_tx=None,
            actuator_monitor=None,
            monitor_settle_sec=0.1,
            feed_state=feed_state,
            dry_run=False,
            allow_feed_without_actuator=False,
        )
        sample = GamepadSample(
            timestamp_sec=1.0,
            axis_x=0.0,
            axis_y=0.0,
            forward_pressed=True,
            backward_pressed=False,
            connected=True,
        )
        loop_state = il2.FeedButtonLoopState()
        before = np.asarray(ctx.sim2real_bridge.motor_target_mm, dtype=np.float64).copy()

        with patch("control.il2._handle_feed_step_once", return_value="sent") as mock_step:
            il2.update_feed_buttons_from_sample(sample, loop_state, ctx, loop_start=1.0)

        mock_step.assert_called_once()
        np.testing.assert_array_equal(ctx.sim2real_bridge.motor_target_mm, before)

    def test_gamepad_sample_hold_pressed_defaults_false(self) -> None:
        sample = GamepadSample(
            timestamp_sec=1.0,
            axis_x=0.0,
            axis_y=0.0,
            forward_pressed=False,
            backward_pressed=False,
            connected=True,
        )

        self.assertFalse(sample.hold_pressed)

    def test_manual_pose_stick_maps_to_direct_motor_target(self) -> None:
        sample = GamepadSample(
            timestamp_sec=1.0,
            axis_x=0.5,
            axis_y=0.25,
            forward_pressed=False,
            backward_pressed=False,
            connected=True,
        )
        state = il2.ManualPoseLoopState()
        cfg = type("Cfg", (), {"invert_yaw": False, "invert_pitch": False})()

        result = il2.compute_manual_pose_result(
            sample=sample,
            teleop_cfg=cfg,
            motor_limit_mm=10.0,
            manual_pose_state=state,
            current_motor_target_mm=np.zeros(4, dtype=np.float64),
        )

        self.assertEqual(result.command, "MANUAL_CMD")
        np.testing.assert_allclose(
            result.motor_target_mm,
            np.asarray([5.0, 2.5, -5.0, -2.5], dtype=np.float64),
        )
        self.assertFalse(result.hold_active)

    def test_manual_pose_stick_center_returns_target_to_zero(self) -> None:
        sample = GamepadSample(
            timestamp_sec=1.0,
            axis_x=0.0,
            axis_y=0.0,
            forward_pressed=False,
            backward_pressed=False,
            connected=True,
        )
        state = il2.ManualPoseLoopState()
        cfg = type("Cfg", (), {"invert_yaw": False, "invert_pitch": False})()

        result = il2.compute_manual_pose_result(
            sample=sample,
            teleop_cfg=cfg,
            motor_limit_mm=10.0,
            manual_pose_state=state,
            current_motor_target_mm=np.asarray([4.0, -3.0, -4.0, 3.0], dtype=np.float64),
        )

        np.testing.assert_allclose(result.motor_target_mm, np.zeros(4, dtype=np.float64))

    def test_manual_pose_b_toggles_hold_and_ignores_stick_until_unlocked(self) -> None:
        cfg = type("Cfg", (), {"invert_yaw": False, "invert_pitch": False})()
        state = il2.ManualPoseLoopState()
        locked_target = np.asarray([3.0, 4.0, -3.0, -4.0], dtype=np.float64)

        lock_sample = GamepadSample(
            timestamp_sec=1.0,
            axis_x=0.0,
            axis_y=0.0,
            forward_pressed=False,
            backward_pressed=False,
            connected=True,
            hold_pressed=True,
        )
        lock_result = il2.compute_manual_pose_result(
            sample=lock_sample,
            teleop_cfg=cfg,
            motor_limit_mm=10.0,
            manual_pose_state=state,
            current_motor_target_mm=locked_target,
        )

        self.assertEqual(lock_result.command, "MANUAL_HOLD")
        np.testing.assert_allclose(lock_result.motor_target_mm, locked_target)
        self.assertTrue(state.hold_active)

        moved_stick_sample = GamepadSample(
            timestamp_sec=1.1,
            axis_x=-1.0,
            axis_y=1.0,
            forward_pressed=False,
            backward_pressed=False,
            connected=True,
            hold_pressed=False,
        )
        held_result = il2.compute_manual_pose_result(
            sample=moved_stick_sample,
            teleop_cfg=cfg,
            motor_limit_mm=10.0,
            manual_pose_state=state,
            current_motor_target_mm=np.asarray([9.0, 9.0, -9.0, -9.0], dtype=np.float64),
        )

        self.assertEqual(held_result.command, "MANUAL_HOLD")
        np.testing.assert_allclose(held_result.motor_target_mm, locked_target)

        unlock_sample = GamepadSample(
            timestamp_sec=1.2,
            axis_x=-1.0,
            axis_y=1.0,
            forward_pressed=False,
            backward_pressed=False,
            connected=True,
            hold_pressed=True,
        )
        unlocked_result = il2.compute_manual_pose_result(
            sample=unlock_sample,
            teleop_cfg=cfg,
            motor_limit_mm=10.0,
            manual_pose_state=state,
            current_motor_target_mm=locked_target,
        )

        self.assertEqual(unlocked_result.command, "MANUAL_CMD")
        np.testing.assert_allclose(
            unlocked_result.motor_target_mm,
            np.asarray([-10.0, 10.0, 10.0, -10.0], dtype=np.float64),
        )
        self.assertFalse(state.hold_active)

    def test_clearing_manual_hold_for_autopilot_keeps_current_bridge_target(self) -> None:
        state = il2.ManualPoseLoopState(
            hold_active=True,
            hold_pressed_prev=True,
            hold_target_mm=np.asarray([2.0, 3.0, -2.0, -3.0], dtype=np.float64),
        )
        bridge_target = np.asarray([2.0, 3.0, -2.0, -3.0], dtype=np.float64)

        il2.clear_manual_pose_hold(state, current_hold_pressed=False)

        self.assertFalse(state.hold_active)
        np.testing.assert_allclose(bridge_target, np.asarray([2.0, 3.0, -2.0, -3.0], dtype=np.float64))

    def test_send_manual_pose_target_clamps_to_motor_limit_without_hardware(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]",
                motor_limit_mm=10.0,
            )
        )
        control = il2.load_yaml_bound_control(yaml_path)
        feed_state = FeedRuntimeState(
            enabled=False,
            cfg=None,
            serial_link=None,
            current_pulses=0,
            locked=False,
            next_send_ts=0.0,
        )
        ctx = TeleopRuntimeContext(
            sim2real_bridge=control.sim2real_bridge,
            motor_mapper=control.motor_mapper,
            actuator_ids=control.actuator_ids,
            actuator_tx=None,
            actuator_monitor=None,
            monitor_settle_sec=0.1,
            feed_state=feed_state,
            dry_run=True,
            allow_feed_without_actuator=False,
        )
        state = il2.ManualPoseLoopState()

        ok = il2.send_manual_pose_target_mm(
            control=control,
            ctx=ctx,
            manual_pose_state=state,
            motor_target_mm=np.asarray([20.0, -20.0, -20.0, 20.0], dtype=np.float64),
            motor_limit_active=(True, True, True, True),
            actuator_tx=None,
        )

        self.assertTrue(ok)
        np.testing.assert_allclose(
            ctx.sim2real_bridge.motor_target_mm,
            np.asarray([10.0, -10.0, -10.0, 10.0], dtype=np.float64),
        )

    def test_estop_latched_blocks_manual_pose_target_update(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]"
            )
        )
        control = il2.load_yaml_bound_control(yaml_path)
        control.sim2real_bridge.estop_latched = True
        control.sim2real_bridge.motor_target_mm = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        feed_state = FeedRuntimeState(
            enabled=False,
            cfg=None,
            serial_link=None,
            current_pulses=0,
            locked=False,
            next_send_ts=0.0,
        )
        ctx = TeleopRuntimeContext(
            sim2real_bridge=control.sim2real_bridge,
            motor_mapper=control.motor_mapper,
            actuator_ids=control.actuator_ids,
            actuator_tx=None,
            actuator_monitor=None,
            monitor_settle_sec=0.1,
            feed_state=feed_state,
            dry_run=True,
            allow_feed_without_actuator=False,
        )

        ok = il2.send_manual_pose_target_mm(
            control=control,
            ctx=ctx,
            manual_pose_state=il2.ManualPoseLoopState(),
            motor_target_mm=np.zeros(4, dtype=np.float64),
            motor_limit_active=(False, False, False, False),
            actuator_tx=None,
        )

        self.assertTrue(ok)
        np.testing.assert_allclose(
            ctx.sim2real_bridge.motor_target_mm,
            np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
        )

    def test_write_metric_row_includes_manual_pose_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = il2.SessionRecorder.create(tmpdir)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            sample = GamepadSample(
                timestamp_sec=1.0,
                axis_x=0.0,
                axis_y=0.0,
                forward_pressed=False,
                backward_pressed=False,
                connected=True,
                hold_pressed=True,
            )

            il2.write_metric_row(
                recorder,
                frame_idx=5,
                timestamp_sec=123.0,
                frame_bgr=frame,
                scope_center=(320, 240),
                lumen_center=(321, 242),
                error_x_px=1,
                error_y_px=2,
                error_norm_px=float(np.hypot(1, 2)),
                inference_ms=8.5,
                status_text="OK",
                control_mode="MANUAL_HOLD",
                autopilot_on=False,
                policy_step_yaw=0.0,
                policy_step_pitch=0.0,
                bridge_command="MANUAL_HOLD",
                estop_latched=False,
                sample=sample,
                feed_delta_pulses=0,
                motor_target_mm=np.asarray([1.0, 2.0, -1.0, -2.0], dtype=np.float64),
                manual_hold_active=True,
            )
            recorder.close()

            with (il2.Path(tmpdir) / "frame_metrics.csv").open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(rows[0]["control_mode"], "MANUAL_HOLD")
        self.assertEqual(rows[0]["hold_pressed"], "1")
        self.assertEqual(rows[0]["manual_hold_active"], "1")

    def test_session_recorder_starts_only_when_autopilot_recording_begins(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder, started = il2.maybe_start_session_recorder(
                None,
                output_dir=tmpdir,
                save_session_artifacts=True,
            )
            self.assertTrue(started)
            self.assertIsNotNone(recorder)
            assert recorder is not None
            self.assertTrue((il2.Path(tmpdir) / "frame_metrics.csv").exists())

            same_recorder, started_again = il2.maybe_start_session_recorder(
                recorder,
                output_dir=tmpdir,
                save_session_artifacts=True,
            )
            self.assertFalse(started_again)
            self.assertIs(same_recorder, recorder)
            recorder.close()

        disabled_recorder, disabled_started = il2.maybe_start_session_recorder(
            None,
            output_dir="",
            save_session_artifacts=False,
        )
        self.assertFalse(disabled_started)
        self.assertIsNone(disabled_recorder)

    def test_should_record_frame_requires_autopilot_and_recorder(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = il2.SessionRecorder.create(tmpdir)
            try:
                self.assertFalse(il2.should_record_frame(False, recorder))
                self.assertFalse(il2.should_record_frame(True, None))
                self.assertTrue(il2.should_record_frame(True, recorder))
            finally:
                recorder.close()

    def test_classify_exception_reason_detects_ram_and_cuda_oom(self) -> None:
        self.assertEqual(
            il2._classify_exception_reason("POLICY_FAIL", MemoryError("cannot allocate")),
            "POLICY_FAIL_RAM_OOM",
        )
        self.assertEqual(
            il2._classify_exception_reason("POLICY_FAIL", RuntimeError("CUDA out of memory while allocating")),
            "POLICY_FAIL_CUDA_OOM",
        )
        self.assertEqual(
            il2._classify_exception_reason("POLICY_FAIL", RuntimeError("generic failure")),
            "POLICY_FAIL",
        )

    def test_build_and_write_exit_diagnostic_report(self) -> None:
        report = il2._build_exit_diagnostic_report(
            "CAMERA_READ_FAIL",
            loop_state={
                "frame_idx": 123,
                "autopilot_on": True,
                "status": "OK",
                "bridge_command": "CMD",
            },
            proc_status={
                "VmRSS": 256 * 1024 * 1024,
                "VmHWM": 384 * 1024 * 1024,
                "VmSize": 1024 * 1024 * 1024,
                "Threads": 12,
            },
            meminfo={
                "MemAvailable": 8 * 1024 * 1024 * 1024,
                "MemFree": 2 * 1024 * 1024 * 1024,
                "MemTotal": 16 * 1024 * 1024 * 1024,
                "SwapFree": 4 * 1024 * 1024 * 1024,
                "SwapTotal": 4 * 1024 * 1024 * 1024,
            },
            torch_cuda={
                "cuda_available": True,
                "device_name": "Test GPU",
                "allocated_bytes": 512 * 1024 * 1024,
                "reserved_bytes": 768 * 1024 * 1024,
                "free_bytes": 6 * 1024 * 1024 * 1024,
                "total_bytes": 8 * 1024 * 1024 * 1024,
                "max_allocated_bytes": 1024 * 1024 * 1024,
            },
            nvidia_smi={
                "name": "Test GPU",
                "used_mib": "512",
                "free_mib": "6144",
                "total_mib": "8192",
            },
        )

        self.assertIn("reason: CAMERA_READ_FAIL", report)
        self.assertIn("frame_idx=123", report)
        self.assertIn("proc_rss=256.00 MiB", report)
        self.assertIn("device=Test GPU", report)

        with TemporaryDirectory() as tmpdir:
            report_path = il2._write_exit_diagnostic_report(report, output_root=il2.Path(tmpdir))

            self.assertTrue(report_path.exists())
            self.assertIn("CAMERA_READ_FAIL", report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
