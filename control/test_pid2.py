import csv
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

import numpy as np

import control.pid2 as pid2
from control.teleop_input import GamepadSample
from control.teleop_runtime import FeedRuntimeState, TeleopRuntimeContext


def build_yaml_text(
    *,
    J_rows: str,
    motor_limit_mm: float = 10.0,
    yaw_limit_deg: float = 170.0,
    pitch_limit_deg: float = 170.0,
    yaw_kp: float = 0.05,
    yaw_ki: float = 0.05,
    yaw_kd: float = 0.0,
    yaw_output_limit_rad: float = 2.1,
    pitch_kp: float = 0.05,
    pitch_ki: float = 0.05,
    pitch_kd: float = 0.0,
    pitch_output_limit_rad: float = 2.1,
    filter_window: int = 5,
    integral_disable_error_abs: float = 0.15,
    large_error_abs: float = 0.2,
    near_center_error_abs: float = 0.05,
    large_error_kp_scale: float = 0.7,
    large_error_kd_scale: float = 1.5,
    near_center_kp_scale: float = 2.0,
    near_center_kd_scale: float = 1.0,
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
  pid:
    filter_window: {filter_window}
    yaw:
      kp: {yaw_kp}
      ki: {yaw_ki}
      kd: {yaw_kd}
      output_limit_rad: {yaw_output_limit_rad}
    pitch:
      kp: {pitch_kp}
      ki: {pitch_ki}
      kd: {pitch_kd}
      output_limit_rad: {pitch_output_limit_rad}
    integral_disable_error_abs: {integral_disable_error_abs}
    large_error_abs: {large_error_abs}
    near_center_error_abs: {near_center_error_abs}
    large_error_kp_scale: {large_error_kp_scale}
    large_error_kd_scale: {large_error_kd_scale}
    near_center_kp_scale: {near_center_kp_scale}
    near_center_kd_scale: {near_center_kd_scale}
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


class PID2Tests(unittest.TestCase):
    def _write_yaml(self, text: str) -> str:
        fp = NamedTemporaryFile("w+", suffix=".yaml", encoding="utf-8", delete=False)
        self.addCleanup(lambda: __import__("os").unlink(fp.name))
        fp.write(text)
        fp.flush()
        fp.close()
        return fp.name

    def test_pid_yaml_parameters_affect_output(self) -> None:
        yaml_a = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]",
                yaw_kp=0.05,
            )
        )
        yaml_b = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]",
                yaw_kp=0.10,
            )
        )

        control_a = pid2.load_yaml_bound_control(yaml_a)
        control_b = pid2.load_yaml_bound_control(yaml_b)
        runtime_a = pid2.PIDRuntimeState.create(control_a.pid_cfg)
        runtime_b = pid2.PIDRuntimeState.create(control_b.pid_cfg)

        filtered_a, _ = runtime_a.update_filtered_error(0.1, 0.0)
        filtered_b, _ = runtime_b.update_filtered_error(0.1, 0.0)
        output_a = runtime_a.yaw_controller.update(filtered_a, 0.0, dt=1.0 / 30.0)
        output_b = runtime_b.yaw_controller.update(filtered_b, 0.0, dt=1.0 / 30.0)

        self.assertNotEqual(output_a, output_b)
        self.assertGreater(abs(output_b), abs(output_a))

    def test_integral_disable_and_recovery(self) -> None:
        pid_cfg = pid2.PIDConfig()
        controller = pid2.FuzzyPIDController(pid_cfg.yaw, pid_cfg)

        controller.update(current_val=0.2, target_val=0.0, dt=0.1)
        self.assertEqual(controller.ki, 0.0)
        self.assertEqual(controller.integral, 0.0)

        controller.update(current_val=0.01, target_val=0.0, dt=0.1)
        self.assertEqual(controller.ki, pid_cfg.yaw.ki)
        self.assertNotEqual(controller.integral, 0.0)

    def test_no_lumen_hold_resets_pid_and_filter_state(self) -> None:
        pid_runtime = pid2.PIDRuntimeState.create(pid2.PIDConfig(filter_window=5))
        pid_runtime.update_filtered_error(0.03, -0.02)
        pid_runtime.yaw_controller.update(0.03, 0.0, dt=0.1)
        pid_runtime.pitch_controller.update(-0.02, 0.0, dt=0.1)

        bridge_command = pid2.handle_no_lumen_hold(pid_runtime)

        self.assertEqual(bridge_command, "NO_LUMEN_HOLD")
        self.assertEqual(len(pid_runtime.history_x), 0)
        self.assertEqual(len(pid_runtime.history_y), 0)
        self.assertEqual(pid_runtime.yaw_controller.integral, 0.0)
        self.assertEqual(pid_runtime.pitch_controller.integral, 0.0)
        self.assertEqual(pid_runtime.yaw_controller.prev_error, 0.0)
        self.assertEqual(pid_runtime.pitch_controller.prev_error, 0.0)
        self.assertEqual(pid_runtime.yaw_controller.last_output, 0.0)
        self.assertEqual(pid_runtime.pitch_controller.last_output, 0.0)

    def test_pid_angle_limit_returns_estop(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [5.0, 0.0]\n    - [0.0, 5.0]\n    - [-5.0, 0.0]\n    - [0.0, -5.0]",
                yaw_limit_deg=1.0,
            )
        )
        control = pid2.load_yaml_bound_control(yaml_path)

        result = pid2.apply_pid_action(control, action=(np.deg2rad(2.0), 0.0), dt=1.0 / 30.0)

        self.assertEqual(result.command, pid2.BridgeCommand.ESTOP)
        self.assertTrue(control.sim2real_bridge.estop_latched)

    def test_motor_target_candidate_is_clamped_by_motor_limit(self) -> None:
        yaml_path = self._write_yaml(
            build_yaml_text(
                J_rows="    - [300.0, 0.0]\n    - [-300.0, 0.0]\n    - [0.0, 300.0]\n    - [0.0, -300.0]",
                motor_limit_mm=10.0,
            )
        )
        control = pid2.load_yaml_bound_control(yaml_path)

        result = pid2.apply_pid_action(control, action=(0.1, 0.1), dt=1.0 / 30.0)

        np.testing.assert_allclose(
            result.motor_target_mm,
            np.array([10.0, -10.0, 10.0, -10.0], dtype=np.float64),
        )

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
        loop_state = pid2.FeedButtonLoopState()
        before = np.asarray(ctx.sim2real_bridge.motor_target_mm, dtype=np.float64).copy()

        with patch("control.pid2._handle_feed_step_once", return_value="sent") as mock_step:
            pid2.update_feed_buttons_from_sample(sample, loop_state, ctx, loop_start=1.0)

        mock_step.assert_called_once()
        np.testing.assert_array_equal(ctx.sim2real_bridge.motor_target_mm, before)

    def test_write_metric_row_includes_pid_fields(self) -> None:
        with TemporaryDirectory() as tmpdir:
            recorder = pid2.SessionRecorder.create(tmpdir)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            sample = GamepadSample(
                timestamp_sec=1.0,
                axis_x=0.0,
                axis_y=0.0,
                forward_pressed=False,
                backward_pressed=True,
                connected=True,
            )

            pid2.write_metric_row(
                recorder,
                frame_idx=7,
                timestamp_sec=123.456,
                frame_bgr=frame,
                scope_center=(320, 240),
                lumen_center=(330, 220),
                error_x_px=10,
                error_y_px=-20,
                error_norm_px=float(np.hypot(10, -20)),
                inference_ms=8.5,
                status_text="OK",
                autopilot_on=True,
                filtered_norm_x=0.02083333,
                filtered_norm_y=-0.04166667,
                controller_step_yaw=0.0125,
                controller_step_pitch=-0.025,
                bridge_command="CMD",
                estop_latched=False,
                sample=sample,
                feed_delta_pulses=60,
                motor_target_mm=np.array([1.0, 2.0, -1.0, -2.0], dtype=np.float64),
            )
            recorder.close()

            with recorder.csv_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["controller_name"], pid2.CONTROLLER_NAME)
            self.assertEqual(row["bridge_command"], "CMD")
            self.assertEqual(row["filtered_norm_x"], "0.02083333")
            self.assertEqual(row["filtered_norm_y"], "-0.04166667")
            self.assertEqual(row["controller_step_yaw"], "0.01250000")
            self.assertEqual(row["controller_step_pitch"], "-0.02500000")

    def test_main_startup_failure_writes_pid2_exit_diagnostics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with patch.object(pid2, "cv2", None), patch.object(
                pid2, "_CV2_IMPORT_ERROR", RuntimeError("missing cv2")
            ), patch("control.pid2.Path.cwd", return_value=Path(tmpdir)):
                status = pid2.main()

            self.assertEqual(status, 1)
            report_path = Path(tmpdir) / pid2.EXIT_DIAGNOSTICS_FILENAME
            self.assertTrue(report_path.exists())
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("STARTUP_CV2_IMPORT_FAIL", text)
            self.assertIn(">>> [PID2] Exit diagnostics", text)


if __name__ == "__main__":
    unittest.main()
