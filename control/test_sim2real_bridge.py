import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from control.sim2real_bridge import (
    BridgeCommand,
    Sim2RealBridge,
    Sim2RealConfig,
    load_dagger_sim2real_runtime_config,
)


class Sim2RealBridgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.J = np.array(
            [
                [2.0, 0.5],
                [-2.0, -0.5],
                [0.1, 1.5],
                [-0.1, -1.5],
            ],
            dtype=np.float64,
        )
        self.bridge = Sim2RealBridge(
            Sim2RealConfig(
                J_4x2_mm_per_rad=self.J,
                yaw_limit_deg=120.0,
                pitch_limit_deg=120.0,
                motor_limit_mm=15.0,
                control_hz=30.0,
            )
        )

    def test_absolute_target_accumulation(self) -> None:
        dyaw1, dpitch1 = 0.2, -0.1
        dyaw2, dpitch2 = -0.05, 0.03

        r1 = self.bridge.step(dyaw1, dpitch1, dt=1.0 / 30.0)
        r2 = self.bridge.step(dyaw2, dpitch2, dt=1.0 / 30.0)

        d1 = self.J @ np.array([dyaw1, dpitch1], dtype=np.float64)
        d2 = self.J @ np.array([dyaw2, dpitch2], dtype=np.float64)

        self.assertEqual(r1.command, BridgeCommand.CMD)
        self.assertTrue(r1.should_send)
        self.assertTrue(r1.serial_frame.startswith("CMD,"))
        np.testing.assert_allclose(r1.motor_delta_mm, d1)
        np.testing.assert_allclose(r1.motor_target_mm, d1)

        self.assertEqual(r2.command, BridgeCommand.CMD)
        self.assertTrue(r2.should_send)
        self.assertTrue(r2.serial_frame.startswith("CMD,"))
        np.testing.assert_allclose(r2.motor_delta_mm, d2)
        np.testing.assert_allclose(r2.motor_target_mm, d1 + d2)

    def test_motor_target_clamp_without_estop(self) -> None:
        bridge = Sim2RealBridge(
            Sim2RealConfig(
                J_4x2_mm_per_rad=np.array(
                    [
                        [300.0, 0.0],
                        [-300.0, 0.0],
                        [0.0, 300.0],
                        [0.0, -300.0],
                    ],
                    dtype=np.float64,
                ),
                yaw_limit_deg=120.0,
                pitch_limit_deg=120.0,
                motor_limit_mm=15.0,
                control_hz=30.0,
            )
        )

        result = bridge.step(0.1, 0.1, dt=1.0 / 30.0)

        self.assertEqual(result.command, BridgeCommand.CMD)
        self.assertFalse(bridge.estop_latched)
        np.testing.assert_allclose(
            result.motor_target_mm,
            np.array([15.0, -15.0, 15.0, -15.0], dtype=np.float64),
        )

    def test_30hz_schedule_on_200hz_sim(self) -> None:
        send_count = 0
        total_steps = 200
        dt = 0.005

        for _ in range(total_steps):
            result = self.bridge.step(0.001, 0.0, dt=dt)
            if result.should_send:
                send_count += 1
                self.assertTrue(result.serial_frame.startswith("CMD,"))
            else:
                self.assertEqual(result.serial_frame, "")

        # 200Hz 仿真跑 1 秒，30Hz 下发应接近 30 次
        self.assertGreaterEqual(send_count, 29)
        self.assertLessEqual(send_count, 31)

    def test_reset_returns_zero_frames(self) -> None:
        self.bridge.step(0.02, -0.01, dt=1.0 / 30.0)
        self.bridge.step(0.01, 0.01, dt=1.0 / 30.0)

        self.bridge.reset()

        self.assertFalse(self.bridge.estop_latched)
        self.assertEqual(self.bridge.yaw_accum_rad, 0.0)
        self.assertEqual(self.bridge.pitch_accum_rad, 0.0)
        self.assertEqual(self.bridge.control_dt_accum, 0.0)
        np.testing.assert_allclose(self.bridge.motor_target_mm, np.zeros(4, dtype=np.float64))

        reset_frame, zero_cmd_frame = self.bridge.get_reset_frames()
        self.assertEqual(reset_frame, "RESET\n")
        self.assertEqual(zero_cmd_frame, "CMD,0.000000,0.000000,0.000000,0.000000\n")

    def test_limit_boundary_and_trip(self) -> None:
        self.bridge.yaw_accum_rad = np.deg2rad(119.9)
        ok_result = self.bridge.step(np.deg2rad(0.1), 0.0, dt=1.0 / 30.0)
        self.assertEqual(ok_result.command, BridgeCommand.CMD)
        self.assertTrue(np.isclose(self.bridge.yaw_accum_rad, np.deg2rad(120.0)))

        trip_result = self.bridge.step(np.deg2rad(1e-3), 0.0, dt=1.0 / 30.0)
        self.assertEqual(trip_result.command, BridgeCommand.ESTOP)
        self.assertTrue(self.bridge.estop_latched)
        self.assertEqual(trip_result.serial_frame, "ESTOP,ANGLE_LIMIT\n")

    def test_yaml_runtime_config_loading(self) -> None:
        yaml_text = """
sim2real:
  J_4x2_mm_per_rad:
    - [3.0, 0.0]
    - [-3.0, 0.0]
    - [0.0, 2.0]
    - [0.0, -2.0]
  yaw_limit_deg: 115.0
  pitch_limit_deg: 110.0
  motor_limit_mm: 12.5
  control_hz: 25.0
  motor_order: ["a", "b", "c", "d"]
  serial:
    port: "/dev/pts/8"
    baudrate: 230400
    timeout: 0.05
  alarm:
    enabled: true
    repeat: 2
    terminal_bell: false
    banner_width: 88
  output:
    print_tx_frame: false
    print_every_n: 50
    save_csv: true
    csv_path: "./tmp_motor.csv"
    save_plot: true
    plot_path: "./tmp_motor.png"
    plot_dpi: 150
    max_plot_points: 1234
"""
        with NamedTemporaryFile("w+", suffix=".yaml", encoding="utf-8") as fp:
            fp.write(yaml_text)
            fp.flush()
            runtime_cfg = load_dagger_sim2real_runtime_config(fp.name)

        np.testing.assert_allclose(
            runtime_cfg.bridge.J_4x2_mm_per_rad,
            np.array([[3.0, 0.0], [-3.0, 0.0], [0.0, 2.0], [0.0, -2.0]], dtype=np.float64),
        )
        self.assertEqual(runtime_cfg.bridge.serial_port, "/dev/pts/8")
        self.assertEqual(runtime_cfg.bridge.serial_baudrate, 230400)
        self.assertAlmostEqual(runtime_cfg.bridge.motor_limit_mm, 12.5)
        self.assertAlmostEqual(runtime_cfg.bridge.control_hz, 25.0)
        self.assertTrue(runtime_cfg.alarm.enabled)
        self.assertEqual(runtime_cfg.alarm.repeat, 2)
        self.assertFalse(runtime_cfg.alarm.terminal_bell)
        self.assertEqual(runtime_cfg.alarm.banner_width, 88)
        self.assertEqual(runtime_cfg.output.print_every_n, 50)
        self.assertEqual(runtime_cfg.output.csv_path, "./tmp_motor.csv")
        self.assertEqual(runtime_cfg.output.plot_path, "./tmp_motor.png")
        self.assertEqual(runtime_cfg.output.plot_dpi, 150)
        self.assertEqual(runtime_cfg.output.max_plot_points, 1234)


if __name__ == "__main__":
    unittest.main()
