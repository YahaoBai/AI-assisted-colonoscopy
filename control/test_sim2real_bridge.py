import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from control.sim2real_bridge import (
    ActuatorConfig,
    BridgeCommand,
    LAFrameBuilder,
    MotorAxisMapConfig,
    MotorMapper,
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
        np.testing.assert_allclose(r1.motor_delta_mm, d1)
        np.testing.assert_allclose(r1.motor_target_mm, d1)

        self.assertEqual(r2.command, BridgeCommand.CMD)
        self.assertTrue(r2.should_send)
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

        # 200Hz 仿真跑 1 秒，30Hz 下发应接近 30 次
        self.assertGreaterEqual(send_count, 29)
        self.assertLessEqual(send_count, 31)

    def test_limit_boundary_and_trip(self) -> None:
        self.bridge.yaw_accum_rad = np.deg2rad(119.9)
        ok_result = self.bridge.step(np.deg2rad(0.1), 0.0, dt=1.0 / 30.0)
        self.assertEqual(ok_result.command, BridgeCommand.CMD)
        self.assertTrue(np.isclose(self.bridge.yaw_accum_rad, np.deg2rad(120.0)))

        trip_result = self.bridge.step(np.deg2rad(1e-3), 0.0, dt=1.0 / 30.0)
        self.assertEqual(trip_result.command, BridgeCommand.ESTOP)
        self.assertTrue(self.bridge.estop_latched)
        self.assertEqual(trip_result.reason, "ANGLE_LIMIT")

    def test_frame_builder_follow_broadcast(self) -> None:
        frame = LAFrameBuilder.build_follow_broadcast_frame(
            actuator_ids=[1, 2, 3, 4],
            target_counts=[1000, 1200, 800, 900],
        )

        body = [
            0x0D,
            0xFF,
            0xF3,
            0x01,
            0xE8,
            0x03,
            0x02,
            0xB0,
            0x04,
            0x03,
            0x20,
            0x03,
            0x04,
            0x84,
            0x03,
        ]
        checksum = sum(body) & 0xFF
        expected = bytes([0x55, 0xAA] + body + [checksum])

        self.assertEqual(frame, expected)

    def test_frame_builder_single_control(self) -> None:
        frame = LAFrameBuilder.build_single_control_frame(actuator_id=3, cmd_value=0x23)
        self.assertEqual(frame, bytes([0x55, 0xAA, 0x03, 0x03, 0x04, 0x00, 0x23, 0x2D]))

    def test_motor_mapper_with_order_and_clamp(self) -> None:
        actuator_cfg = ActuatorConfig(
            mode="broadcast_follow_no_feedback",
            ids=(1, 2, 3, 4),
            position_index=0x37,
            count_min=0,
            count_max=2000,
            m1=MotorAxisMapConfig(zero_count=1000, count_per_mm=100.0, sign=1, soft_min_count=300, soft_max_count=1700),
            m2=MotorAxisMapConfig(zero_count=1000, count_per_mm=100.0, sign=-1, soft_min_count=300, soft_max_count=1700),
            m3=MotorAxisMapConfig(zero_count=1000, count_per_mm=100.0, sign=1, soft_min_count=300, soft_max_count=1700),
            m4=MotorAxisMapConfig(zero_count=1000, count_per_mm=100.0, sign=1, soft_min_count=300, soft_max_count=1700),
        )
        mapper = MotorMapper(actuator_cfg, motor_order=("m2", "m1", "m4", "m3"))

        mm_targets = np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float64)  # m1,m2,m3,m4
        counts = mapper.mm_targets_to_counts(mm_targets)

        np.testing.assert_array_equal(counts, np.array([1100, 1100, 800, 1200], dtype=np.int32))

    def test_actuator_id_mapping_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ActuatorConfig(
                mode="broadcast_follow_no_feedback",
                ids=(1, 2, 3, 4),
                id_by_motor={"m1": 1, "m2": 3, "m3": 2, "m4": 4},
                position_index=0x37,
                count_min=0,
                count_max=2000,
            )

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
  motor_order: ["m2", "m1", "m4", "m3"]
  serial:
    port: "/dev/pts/8"
    baudrate: 230400
    timeout: 0.05
    write_timeout: 0.12
    critical_retry_count: 5
    critical_retry_interval_sec: 0.03
  actuator:
    mode: broadcast_follow_no_feedback
    ids: [4, 3, 2, 1]
    id_by_motor:
      m1: 4
      m2: 3
      m3: 2
      m4: 1
    position_index: 0x37
    count_min: 0
    count_max: 2000
    per_motor:
      m1:
        zero_count: 1001
        count_per_mm: 10.0
        sign: 1
        soft_min_count: 200
        soft_max_count: 1800
      m2:
        zero_count: 999
        count_per_mm: 11.0
        sign: -1
        soft_min_count: 250
        soft_max_count: 1750
      m3:
        zero_count: 1000
        count_per_mm: 12.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
      m4:
        zero_count: 1000
        count_per_mm: 13.0
        sign: 1
        soft_min_count: 350
        soft_max_count: 1650
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
        self.assertAlmostEqual(runtime_cfg.bridge.serial_write_timeout, 0.12)
        self.assertEqual(runtime_cfg.bridge.serial_critical_retry_count, 5)
        self.assertAlmostEqual(runtime_cfg.bridge.serial_critical_retry_interval_sec, 0.03)
        self.assertAlmostEqual(runtime_cfg.bridge.motor_limit_mm, 12.5)
        self.assertAlmostEqual(runtime_cfg.bridge.control_hz, 25.0)
        self.assertEqual(runtime_cfg.bridge.motor_order, ("m2", "m1", "m4", "m3"))
        self.assertEqual(runtime_cfg.actuator.ids, (4, 3, 2, 1))
        self.assertEqual(runtime_cfg.actuator.id_by_motor["m3"], 2)
        self.assertEqual(
            runtime_cfg.actuator.ids_for_motor_order(("m2", "m1", "m4", "m3")),
            (3, 4, 1, 2),
        )
        self.assertEqual(runtime_cfg.actuator.position_index, 0x37)
        self.assertEqual(runtime_cfg.actuator.m2.sign, -1)
        self.assertAlmostEqual(runtime_cfg.actuator.m4.count_per_mm, 13.0)
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
