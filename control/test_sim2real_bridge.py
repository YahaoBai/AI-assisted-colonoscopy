import unittest
from tempfile import NamedTemporaryFile
import threading
import time
from unittest.mock import patch

import numpy as np

from control.sim2real_bridge import (
    ActuatorStatus,
    ActuatorMonitor,
    ActuatorConfig,
    ActuatorTx,
    BridgeCommand,
    LAFrameParser,
    LAFrameBuilder,
    MonitorConfig,
    MotorAxisMapConfig,
    MotorMapper,
    Sim2RealBridge,
    Sim2RealConfig,
    load_dagger_sim2real_runtime_config,
)


def build_status_response_frame(
    actuator_id: int,
    target_count: int,
    current_count: int,
    temperature_c: int,
    error_bits: int,
    motor_current_ma: int = 100,
    force_sensor_raw: int = 0,
    internal_data_1: int = 0,
    internal_data_2: int = 0,
) -> bytes:
    force_sensor = int(force_sensor_raw) & 0xFFFF
    data = [
        target_count & 0xFF,
        (target_count >> 8) & 0xFF,
        current_count & 0xFF,
        (current_count >> 8) & 0xFF,
        temperature_c & 0xFF,
        motor_current_ma & 0xFF,
        (motor_current_ma >> 8) & 0xFF,
        force_sensor & 0xFF,
        error_bits & 0xFF,
        (force_sensor >> 8) & 0xFF,
        internal_data_1 & 0xFF,
        (internal_data_1 >> 8) & 0xFF,
        internal_data_2 & 0xFF,
        (internal_data_2 >> 8) & 0xFF,
    ]
    frame_len = len(data) + 3
    body = [frame_len, actuator_id, 0x04, 0x00, 0x22] + data
    checksum = sum(body) & 0xFF
    return bytes([0xAA, 0x55] + body + [checksum])


class MockSerialForMonitor:
    def __init__(self, response_by_id):
        self.response_by_id = response_by_id
        self.is_open = True
        self._rx = bytearray()
        self._io_lock = threading.Lock()
        self.write_history = []

    @property
    def in_waiting(self) -> int:
        with self._io_lock:
            return len(self._rx)

    def reset_input_buffer(self) -> None:
        with self._io_lock:
            self._rx.clear()

    def write(self, payload: bytes) -> int:
        frame = bytes(payload)
        self.write_history.append(frame)

        if len(frame) >= 8 and frame[0:2] == b"\x55\xAA" and frame[4] == 0x04 and frame[6] == 0x22:
            actuator_id = int(frame[3])
            response = self.response_by_id.get(actuator_id)
            if callable(response):
                response = response(actuator_id)
            if response:
                with self._io_lock:
                    self._rx.extend(bytes(response))
        return len(frame)

    def flush(self) -> None:
        return None

    def read(self, size: int = 1) -> bytes:
        with self._io_lock:
            if len(self._rx) == 0:
                return b""
            n = max(1, min(int(size), len(self._rx)))
            out = bytes(self._rx[:n])
            del self._rx[:n]
            return out


class Sim2RealBridgeTests(unittest.TestCase):
    def _wait_until(self, predicate, timeout_sec: float = 0.5, interval_sec: float = 0.005) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(interval_sec)
        return bool(predicate())

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

    def test_actuator_tx_fault_clear_all(self) -> None:
        serial_mock = MockSerialForMonitor(response_by_id={})
        actuator_tx = ActuatorTx(serial_mock)

        ok = actuator_tx.send_fault_clear_all((1, 2, 3, 4), critical=True)

        self.assertTrue(ok)
        self.assertEqual(
            serial_mock.write_history,
            [
                bytes([0x55, 0xAA, 0x03, 0x01, 0x04, 0x00, 0x1E, 0x26]),
                bytes([0x55, 0xAA, 0x03, 0x02, 0x04, 0x00, 0x1E, 0x27]),
                bytes([0x55, 0xAA, 0x03, 0x03, 0x04, 0x00, 0x1E, 0x28]),
                bytes([0x55, 0xAA, 0x03, 0x04, 0x04, 0x00, 0x1E, 0x29]),
            ],
        )

    def test_frame_builder_status_query(self) -> None:
        frame = LAFrameBuilder.build_status_query_frame(actuator_id=3)
        self.assertEqual(frame, bytes([0x55, 0xAA, 0x03, 0x03, 0x04, 0x00, 0x22, 0x2C]))

    def test_frame_parser_status_response(self) -> None:
        frame = build_status_response_frame(
            actuator_id=2,
            target_count=1000,
            current_count=990,
            temperature_c=20,
            error_bits=0x00,
        )
        status = LAFrameParser.parse_status_response(frame, expected_id=2)

        self.assertEqual(status.actuator_id, 2)
        self.assertEqual(status.target_count, 1000)
        self.assertEqual(status.current_count, 990)
        self.assertEqual(status.temperature_c, 20)
        self.assertEqual(status.error_bits, 0x00)

    def test_frame_parser_status_response_reads_error_bits_from_b15(self) -> None:
        frame = build_status_response_frame(
            actuator_id=2,
            target_count=1000,
            current_count=990,
            temperature_c=20,
            error_bits=0x00,
            internal_data_2=0x7F00,
        )
        status = LAFrameParser.parse_status_response(frame, expected_id=2)

        self.assertEqual(status.error_bits, 0x00)

    def test_frame_parser_status_response_bad_checksum(self) -> None:
        frame = bytearray(
            build_status_response_frame(
                actuator_id=1,
                target_count=1000,
                current_count=1000,
                temperature_c=22,
                error_bits=0x00,
            )
        )
        frame[-1] = (frame[-1] + 1) & 0xFF
        with self.assertRaises(ValueError):
            LAFrameParser.parse_status_response(bytes(frame), expected_id=1)

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

    def test_monitor_round_robin_query(self) -> None:
        response_by_id = {
            1: build_status_response_frame(1, 1000, 1000, 20, 0),
            2: build_status_response_frame(2, 1000, 1000, 20, 0),
            3: build_status_response_frame(3, 1000, 1000, 20, 0),
            4: build_status_response_frame(4, 1000, 1000, 20, 0),
        }
        serial_mock = MockSerialForMonitor(response_by_id)
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.01,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        try:
            monitor.start()
            monitor.set_active(True)
            ok = self._wait_until(lambda: len(serial_mock.write_history) >= 4, timeout_sec=0.5)
            self.assertTrue(ok)
            queried_ids = [frame[3] for frame in serial_mock.write_history[:4]]
            self.assertEqual(queried_ids, [1, 2, 3, 4])
            self.assertFalse(monitor.has_fault())
        finally:
            monitor.set_active(False)
            monitor.stop(join_timeout_sec=0.5)

    def test_monitor_logs_four_axis_summary_once_per_round_interval(self) -> None:
        monitor = ActuatorMonitor(
            serial_link=None,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.01,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=1,
            ),
        )

        statuses = [
            ActuatorStatus(
                actuator_id=aid,
                target_count=600 + aid,
                current_count=603 + aid,
                temperature_c=29 + aid,
                error_bits=0x00,
                raw_frame=b"",
                timestamp_sec=1.0 + aid,
            )
            for aid in (1, 2, 3, 4)
        ]

        with patch("builtins.print") as print_mock:
            for status in statuses:
                monitor._record_success(status)

        print_mock.assert_called_once()
        message = print_mock.call_args[0][0]
        self.assertIn("round=1", message)
        self.assertIn("poll=4", message)
        for aid in (1, 2, 3, 4):
            self.assertIn(f"id={aid}", message)

    def test_critical_send_not_interleaved_by_monitor_queries(self) -> None:
        class SlowSerial(MockSerialForMonitor):
            def write(self, payload: bytes) -> int:
                written = super().write(payload)
                time.sleep(0.002)
                return written

        response_by_id = {
            1: build_status_response_frame(1, 1000, 1000, 20, 0),
            2: build_status_response_frame(2, 1000, 1000, 20, 0),
            3: build_status_response_frame(3, 1000, 1000, 20, 0),
            4: build_status_response_frame(4, 1000, 1000, 20, 0),
        }
        serial_mock = SlowSerial(response_by_id)
        actuator_tx = ActuatorTx(serial_mock)
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=500.0,
                response_timeout_sec=0.005,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
            serial_lock=actuator_tx.serial_lock,
        )

        try:
            monitor.start()
            monitor.set_active(True)
            ok = self._wait_until(lambda: len(serial_mock.write_history) >= 4, timeout_sec=0.5)
            self.assertTrue(ok)
            serial_mock.write_history.clear()

            tx_thread = threading.Thread(
                target=lambda: actuator_tx.send_estop_all((1, 2, 3, 4), critical=True)
            )
            tx_thread.start()
            tx_thread.join(timeout=1.0)
            self.assertFalse(tx_thread.is_alive())

            history = list(serial_mock.write_history)
            estop_start = next(
                idx
                for idx, frame in enumerate(history)
                if len(frame) >= 8 and frame[0:2] == b"\x55\xAA" and frame[6] == 0x23
            )
            estop_frames = history[estop_start : estop_start + 4]
            self.assertEqual([frame[3] for frame in estop_frames], [1, 2, 3, 4])
            self.assertTrue(all(frame[6] == 0x23 for frame in estop_frames))
        finally:
            monitor.set_active(False)
            monitor.stop(join_timeout_sec=0.5)

    def test_monitor_failure_threshold_trip(self) -> None:
        serial_mock = MockSerialForMonitor(response_by_id={})
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.003,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        try:
            monitor.start()
            monitor.set_active(True)
            ok = self._wait_until(monitor.has_fault, timeout_sec=0.6)
            self.assertTrue(ok)
            fault = monitor.get_fault()
            self.assertIsNotNone(fault)
            self.assertEqual(fault.consecutive_failures, 3)
            self.assertIn("READ_TIMEOUT", fault.reason)
        finally:
            monitor.set_active(False)
            monitor.stop(join_timeout_sec=0.5)

    def test_monitor_query_temporarily_forces_nonblocking_serial_timeout(self) -> None:
        class TimeoutAwareSerial:
            def __init__(self) -> None:
                self.is_open = True
                self.timeout = 0.05
                self.read_timeouts = []

            @property
            def in_waiting(self) -> int:
                return 0

            def reset_input_buffer(self) -> None:
                return None

            def write(self, payload: bytes) -> int:
                return len(payload)

            def flush(self) -> None:
                return None

            def read(self, size: int = 1) -> bytes:
                self.read_timeouts.append(self.timeout)
                if self.timeout and self.timeout > 0.0:
                    time.sleep(float(self.timeout))
                return b""

        serial_mock = TimeoutAwareSerial()
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.003,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        start = time.monotonic()
        ok, status, reason, _ = monitor._query_once(1)
        elapsed = time.monotonic() - start

        self.assertFalse(ok)
        self.assertIsNone(status)
        self.assertEqual(reason, "READ_TIMEOUT")
        self.assertLess(elapsed, 0.02)
        self.assertGreater(len(serial_mock.read_timeouts), 0)
        self.assertTrue(all(timeout == 0.0 for timeout in serial_mock.read_timeouts))
        self.assertEqual(serial_mock.timeout, 0.05)

    def test_monitor_query_skips_stale_frame_from_previous_axis(self) -> None:
        class StaleResponseSerial:
            def __init__(self) -> None:
                self.is_open = True
                self.timeout = 0.0
                self._rx = bytearray()
                self._stage = 0

            @property
            def in_waiting(self) -> int:
                return len(self._rx)

            def reset_input_buffer(self) -> None:
                self._rx.clear()

            def write(self, payload: bytes) -> int:
                actuator_id = int(payload[3])
                if self._stage == 0 and actuator_id == 1:
                    self._stage = 1
                elif self._stage == 1 and actuator_id == 2:
                    self._rx.extend(build_status_response_frame(1, 1000, 1000, 20, 0x00))
                    self._rx.extend(build_status_response_frame(2, 1002, 1001, 21, 0x00))
                    self._stage = 2
                return len(payload)

            def flush(self) -> None:
                return None

            def read(self, size: int = 1) -> bytes:
                if len(self._rx) == 0:
                    return b""
                n = max(1, min(int(size), len(self._rx)))
                out = bytes(self._rx[:n])
                del self._rx[:n]
                return out

        serial_mock = StaleResponseSerial()
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.003,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        ok, status, reason, _ = monitor._query_once(1)
        self.assertFalse(ok)
        self.assertIsNone(status)
        self.assertEqual(reason, "READ_TIMEOUT")

        ok, status, reason, _ = monitor._query_once(2)
        self.assertTrue(ok)
        self.assertIsNotNone(status)
        self.assertEqual(reason, "OK")
        self.assertEqual(status.actuator_id, 2)
        self.assertEqual(status.target_count, 1002)

    def test_monitor_query_skips_malformed_frame_before_valid_response(self) -> None:
        class MalformedThenValidSerial:
            def __init__(self) -> None:
                self.is_open = True
                self.timeout = 0.0
                self._rx = bytearray()

            @property
            def in_waiting(self) -> int:
                return len(self._rx)

            def reset_input_buffer(self) -> None:
                self._rx.clear()

            def write(self, payload: bytes) -> int:
                if len(self._rx) == 0:
                    malformed = bytearray(build_status_response_frame(1, 1000, 1000, 20, 0x00))
                    malformed[-1] = (malformed[-1] + 1) & 0xFF
                    self._rx.extend(bytes(malformed))
                    self._rx.extend(build_status_response_frame(1, 1003, 1001, 21, 0x00))
                return len(payload)

            def flush(self) -> None:
                return None

            def read(self, size: int = 1) -> bytes:
                if len(self._rx) == 0:
                    return b""
                n = max(1, min(int(size), len(self._rx)))
                out = bytes(self._rx[:n])
                del self._rx[:n]
                return out

        serial_mock = MalformedThenValidSerial()
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1,),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.01,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        ok, status, reason, _ = monitor._query_once(1)

        self.assertTrue(ok)
        self.assertIsNotNone(status)
        self.assertEqual(reason, "OK")
        self.assertEqual(status.actuator_id, 1)
        self.assertEqual(status.target_count, 1003)

    def test_monitor_verify_fault_clear_short_circuits_when_disabled(self) -> None:
        serial_mock = MockSerialForMonitor(response_by_id={})
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=False,
                query_hz=200.0,
                response_timeout_sec=0.01,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        result = monitor.verify_fault_clear(actuator_ids=(1, 2, 3, 4), attempts=1)

        self.assertTrue(result.ok)
        self.assertEqual(result.statuses_by_id, {})
        self.assertEqual(result.failures_by_id, {})
        self.assertEqual(result.uncleared_error_bits_by_id, {})
        self.assertEqual(serial_mock.write_history, [])

    def test_monitor_pause_clears_failure_streaks(self) -> None:
        response_by_id = {
            1: None,
            2: build_status_response_frame(2, 1000, 1000, 20, 0x00),
            3: build_status_response_frame(3, 1000, 1000, 20, 0x00),
            4: build_status_response_frame(4, 1000, 1000, 20, 0x00),
        }
        serial_mock = MockSerialForMonitor(response_by_id=response_by_id)
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.003,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        try:
            monitor.start()
            monitor.set_active(True)
            ok = self._wait_until(
                lambda: monitor.snapshot()["consecutive_failures_by_id"].get(1, 0) >= 2,
                timeout_sec=0.6,
            )
            self.assertTrue(ok)
            self.assertFalse(monitor.has_fault())

            monitor.set_active(False)
            ok = self._wait_until(
                lambda: monitor.snapshot()["consecutive_failures_by_id"].get(1, -1) == 0,
                timeout_sec=0.2,
            )
            self.assertTrue(ok)

            monitor.set_active(True)
            ok = self._wait_until(
                lambda: monitor.snapshot()["consecutive_failures_by_id"].get(1, 0) >= 1,
                timeout_sec=0.3,
            )
            self.assertTrue(ok)
            self.assertFalse(monitor.has_fault())
            self.assertLess(
                monitor.snapshot()["consecutive_failures_by_id"].get(1, 0),
                monitor.monitor_cfg.failure_threshold,
            )
        finally:
            monitor.set_active(False)
            monitor.stop(join_timeout_sec=0.5)

    def test_monitor_failure_threshold_trip_when_one_axis_keeps_timing_out(self) -> None:
        response_by_id = {
            1: None,
            2: build_status_response_frame(2, 1000, 1000, 20, 0x00),
            3: build_status_response_frame(3, 1000, 1000, 20, 0x00),
            4: build_status_response_frame(4, 1000, 1000, 20, 0x00),
        }
        serial_mock = MockSerialForMonitor(response_by_id=response_by_id)
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.003,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        try:
            monitor.start()
            monitor.set_active(True)
            ok = self._wait_until(monitor.has_fault, timeout_sec=0.6)
            self.assertTrue(ok)
            fault = monitor.get_fault()
            self.assertIsNotNone(fault)
            self.assertEqual(fault.actuator_id, 1)
            self.assertEqual(fault.consecutive_failures, 3)
            self.assertIn("READ_TIMEOUT", fault.reason)
        finally:
            monitor.set_active(False)
            monitor.stop(join_timeout_sec=0.5)

    def test_monitor_error_bits_trip(self) -> None:
        response_by_id = {
            1: build_status_response_frame(1, 1000, 1000, 20, 0x02),
            2: build_status_response_frame(2, 1000, 1000, 20, 0x00),
            3: build_status_response_frame(3, 1000, 1000, 20, 0x00),
            4: build_status_response_frame(4, 1000, 1000, 20, 0x00),
        }
        serial_mock = MockSerialForMonitor(response_by_id=response_by_id)
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.01,
                failure_threshold=5,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        try:
            monitor.start()
            monitor.set_active(True)
            ok = self._wait_until(monitor.has_fault, timeout_sec=0.5)
            self.assertTrue(ok)
            fault = monitor.get_fault()
            self.assertIsNotNone(fault)
            self.assertEqual(fault.reason, "ACTUATOR_ERROR_BITS")
            self.assertEqual(fault.actuator_id, 1)
            self.assertEqual(fault.error_bits, 0x02)
        finally:
            monitor.set_active(False)
            monitor.stop(join_timeout_sec=0.5)

    def test_monitor_verify_fault_clear_rejects_uncleared_error_bits(self) -> None:
        response_by_id = {
            1: build_status_response_frame(1, 1000, 1000, 78, 0x01),
            2: build_status_response_frame(2, 1000, 1000, 20, 0x00),
            3: build_status_response_frame(3, 1000, 1000, 20, 0x00),
            4: build_status_response_frame(4, 1000, 1000, 20, 0x00),
        }
        serial_mock = MockSerialForMonitor(response_by_id=response_by_id)
        monitor = ActuatorMonitor(
            serial_link=serial_mock,
            actuator_ids=(1, 2, 3, 4),
            monitor_cfg=MonitorConfig(
                enabled=True,
                query_hz=200.0,
                response_timeout_sec=0.01,
                failure_threshold=3,
                error_mask=0x0F,
                log_every_n=0,
            ),
        )

        result = monitor.verify_fault_clear(actuator_ids=(1, 2, 3, 4), attempts=1)

        self.assertFalse(result.ok)
        self.assertEqual(result.failures_by_id, {})
        self.assertEqual(result.uncleared_error_bits_by_id, {1: 0x01})
        self.assertEqual(result.statuses_by_id[1].temperature_c, 78)
        self.assertFalse(monitor.has_fault())

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
  monitor:
    enabled: true
    query_hz: 18.0
    response_timeout_sec: 0.05
    failure_threshold: 4
    error_mask: 0x0F
    log_every_n: 66
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
        self.assertTrue(runtime_cfg.monitor.enabled)
        self.assertAlmostEqual(runtime_cfg.monitor.query_hz, 18.0)
        self.assertAlmostEqual(runtime_cfg.monitor.response_timeout_sec, 0.05)
        self.assertEqual(runtime_cfg.monitor.failure_threshold, 4)
        self.assertEqual(runtime_cfg.monitor.error_mask, 0x0F)
        self.assertEqual(runtime_cfg.monitor.log_every_n, 66)
        self.assertEqual(runtime_cfg.output.print_every_n, 50)
        self.assertEqual(runtime_cfg.output.csv_path, "./tmp_motor.csv")
        self.assertEqual(runtime_cfg.output.plot_path, "./tmp_motor.png")
        self.assertEqual(runtime_cfg.output.plot_dpi, 150)
        self.assertEqual(runtime_cfg.output.max_plot_points, 1234)

    def test_yaml_runtime_config_rejects_missing_bridge_limit(self) -> None:
        yaml_text = """
sim2real:
  J_4x2_mm_per_rad:
    - [1.0, 0.0]
    - [-1.0, 0.0]
    - [0.0, 1.0]
    - [0.0, -1.0]
  yaw_limit_deg: 120.0
  pitch_limit_deg: 120.0
  control_hz: 30.0
  motor_order: ["m1", "m2", "m3", "m4"]
  serial:
    port: "/dev/ttyUSB0"
    baudrate: 115200
    timeout: 0.0
    write_timeout: 0.2
    critical_retry_count: 3
    critical_retry_interval_sec: 0.02
  actuator:
    mode: broadcast_follow_no_feedback
    ids: [1, 2, 3, 4]
    id_by_motor:
      m1: 1
      m2: 2
      m3: 3
      m4: 4
    position_index: 0x37
    count_min: 0
    count_max: 2000
    per_motor:
      m1:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
      m2:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
      m3:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
      m4:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
  alarm:
    enabled: true
    repeat: 3
    terminal_bell: true
    banner_width: 70
  monitor:
    enabled: true
    query_hz: 20.0
    response_timeout_sec: 0.02
    failure_threshold: 3
    error_mask: 0x0F
    log_every_n: 5
  output:
    print_tx_frame: false
    print_every_n: 0
    save_csv: true
    csv_path: "./sim2real_motor_log.csv"
    save_plot: true
    plot_path: "./sim2real_motor_plot.png"
    plot_dpi: 120
    max_plot_points: 4000
"""
        with NamedTemporaryFile("w+", suffix=".yaml", encoding="utf-8") as fp:
            fp.write(yaml_text)
            fp.flush()
            with self.assertRaisesRegex(ValueError, "sim2real.motor_limit_mm"):
                load_dagger_sim2real_runtime_config(fp.name)

    def test_yaml_runtime_config_rejects_missing_serial_write_timeout(self) -> None:
        yaml_text = """
sim2real:
  J_4x2_mm_per_rad:
    - [1.0, 0.0]
    - [-1.0, 0.0]
    - [0.0, 1.0]
    - [0.0, -1.0]
  yaw_limit_deg: 120.0
  pitch_limit_deg: 120.0
  motor_limit_mm: 10.0
  control_hz: 30.0
  motor_order: ["m1", "m2", "m3", "m4"]
  serial:
    port: "/dev/ttyUSB0"
    baudrate: 115200
    timeout: 0.0
    critical_retry_count: 3
    critical_retry_interval_sec: 0.02
  actuator:
    mode: broadcast_follow_no_feedback
    ids: [1, 2, 3, 4]
    id_by_motor:
      m1: 1
      m2: 2
      m3: 3
      m4: 4
    position_index: 0x37
    count_min: 0
    count_max: 2000
    per_motor:
      m1:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
      m2:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
      m3:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
      m4:
        zero_count: 1000
        count_per_mm: 40.0
        sign: 1
        soft_min_count: 300
        soft_max_count: 1700
  alarm:
    enabled: true
    repeat: 3
    terminal_bell: true
    banner_width: 70
  monitor:
    enabled: true
    query_hz: 20.0
    response_timeout_sec: 0.02
    failure_threshold: 3
    error_mask: 0x0F
    log_every_n: 5
  output:
    print_tx_frame: false
    print_every_n: 0
    save_csv: true
    csv_path: "./sim2real_motor_log.csv"
    save_plot: true
    plot_path: "./sim2real_motor_plot.png"
    plot_dpi: 120
    max_plot_points: 4000
"""
        with NamedTemporaryFile("w+", suffix=".yaml", encoding="utf-8") as fp:
            fp.write(yaml_text)
            fp.flush()
            with self.assertRaisesRegex(ValueError, "sim2real.serial.write_timeout"):
                load_dagger_sim2real_runtime_config(fp.name)


if __name__ == "__main__":
    unittest.main()
