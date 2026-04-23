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
