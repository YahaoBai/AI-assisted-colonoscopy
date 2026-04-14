import os
import tempfile
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import control.teleop_runtime as tr
from control.feed import FeedConfig
from control.sim2real_bridge import BridgeCommand, FaultClearVerification
from control.teleop import (
    HoldLatch,
    PygameGamepadInput,
    TeleopConfig,
    build_arg_parser,
    load_teleop_config,
    perform_reset_sequence,
    should_send_follow_command,
    should_trigger_disconnect_estop,
)


class FakeMapper:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs

    def mm_targets_to_counts(self, mm_targets: np.ndarray) -> np.ndarray:
        arr = np.asarray(mm_targets, dtype=np.float64).reshape(-1)
        if arr.shape != (4,):
            raise ValueError("shape mismatch")
        return np.asarray([1000, 1001, 1002, 1003], dtype=np.int32)


class FakeTx:
    def __init__(
        self,
        calls,
        fault_clear_ok=True,
        work_start_ok=True,
        follow_ok=True,
    ):
        self.calls = calls
        self.fault_clear_ok = fault_clear_ok
        self.work_start_ok = work_start_ok
        self.follow_ok = follow_ok

    def send_fault_clear_all(self, actuator_ids, critical=True):
        self.calls.append(("fault_clear", tuple(actuator_ids), bool(critical)))
        return self.fault_clear_ok

    def send_work_start_all(self, actuator_ids, critical=True):
        self.calls.append(("work_start", tuple(actuator_ids), bool(critical)))
        return self.work_start_ok

    def send_follow_broadcast(self, actuator_ids, target_counts, critical=False, frame_name="F3_FOLLOW"):
        self.calls.append(
            (
                "follow",
                tuple(actuator_ids),
                tuple(int(x) for x in target_counts),
                bool(critical),
                str(frame_name),
            )
        )
        return self.follow_ok


class FakeMonitor:
    def __init__(self, verify_ok=True):
        self.monitor_cfg = SimpleNamespace(enabled=True)
        self.verify_ok = bool(verify_ok)
        self.clear_fault_called = False
        self.verify_calls = 0

    def verify_fault_clear(self, actuator_ids, attempts, settle_time_sec):
        self.verify_calls += 1
        if self.verify_ok:
            return FaultClearVerification(ok=True)
        return FaultClearVerification(
            ok=False,
            failures_by_id={1: "READ_TIMEOUT"},
            uncleared_error_bits_by_id={2: 0x02},
        )

    def clear_fault(self):
        self.clear_fault_called = True


class FakeBridge:
    def __init__(self):
        self.reset_called = 0
        self.estop_latched = False

    def reset(self):
        self.reset_called += 1

    def step(self, delta_yaw: float, delta_pitch: float, dt: float):
        return SimpleNamespace(
            command=BridgeCommand.CMD,
            should_send=False,
            motor_target_mm=np.zeros(4, dtype=np.float64),
            reason="",
        )


class FakeFeedSerial:
    def __init__(self):
        self.writes = []
        self.closed = False

    def write(self, data: bytes) -> int:
        self.writes.append(bytes(data))
        return len(data)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeRuntimeBridge:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs
        self.estop_latched = False
        self.motor_target_mm = np.zeros(4, dtype=np.float64)
        self.motor_limit_mm = 8.0
        self.reset_called = 0

    def reset(self) -> None:
        self.reset_called += 1

    def step(self, delta_yaw: float, delta_pitch: float, dt: float):
        _ = delta_yaw
        _ = delta_pitch
        _ = dt
        return SimpleNamespace(
            command=BridgeCommand.CMD,
            should_send=False,
            motor_target_mm=np.zeros(4, dtype=np.float64),
            reason="",
        )


class FakeRuntimeGamepad:
    def __init__(self, cfg):
        _ = cfg
        self.controller_index = 0
        self.controller_name = "fake-gamepad"
        self._poll_count = 0

    def list_controllers(self):
        return [(0, "fake-gamepad", True)]

    def open(self) -> None:
        return None

    def poll(self):
        self._poll_count += 1
        if self._poll_count == 1:
            return SimpleNamespace(
                timestamp_sec=0.0,
                axis_x=0.7,
                axis_y=-0.7,
                forward_pressed=False,
                estop_pressed=False,
                reset_primary_pressed=False,
                reset_secondary_pressed=False,
                connected=True,
            )
        raise KeyboardInterrupt()

    def close(self) -> None:
        return None


class TeleopCoreTests(unittest.TestCase):
    @staticmethod
    def _make_feed_state(
        *,
        teleop_enabled: bool = True,
        dry_run: bool = True,
        locked: bool = False,
        repeat_hz: float = 10.0,
    ) -> tr.FeedRuntimeState:
        cfg = FeedConfig(
            port="/dev/ttyUSB1",
            baudrate=115200,
            timeout=0.05,
            addr=1,
            step_pulses=200,
            repeat_hz=repeat_hz,
            default_vel=100,
            default_acc=0,
            invert_dir=False,
            min_pulses=0,
            max_pulses=1000,
            enable_on_start=True,
            disable_on_exit=True,
            teleop_enabled=teleop_enabled,
            dry_run=dry_run,
        )
        cfg.validate()
        return tr.FeedRuntimeState(
            enabled=bool(teleop_enabled),
            cfg=cfg,
            serial_link=FakeFeedSerial() if (teleop_enabled and (not dry_run)) else None,
            current_pulses=0,
            locked=bool(locked),
        )

    @staticmethod
    def _make_ctx(feed_state: tr.FeedRuntimeState) -> tr.TeleopRuntimeContext:
        return tr.TeleopRuntimeContext(
            sim2real_bridge=FakeBridge(),
            motor_mapper=FakeMapper(),
            actuator_ids=(1, 2, 3, 4),
            actuator_tx=None,
            actuator_monitor=None,
            monitor_settle_sec=0.02,
            feed_state=feed_state,
            dry_run=True,
            allow_feed_without_actuator=False,
        )

    @staticmethod
    def _make_runtime_cfg():
        class _ActuatorCfg:
            @staticmethod
            def ids_for_motor_order(_order):
                return (1, 2, 3, 4)

        return SimpleNamespace(
            bridge=SimpleNamespace(
                serial_port="/dev/ttyUSB0",
                serial_baudrate=115200,
                serial_timeout=0.0,
                serial_write_timeout=0.2,
                serial_critical_retry_count=3,
                serial_critical_retry_interval_sec=0.02,
                control_hz=30.0,
                motor_order=("m1", "m2", "m3", "m4"),
            ),
            actuator=_ActuatorCfg(),
            monitor=SimpleNamespace(response_timeout_sec=0.02),
            output=SimpleNamespace(print_tx_frame=False),
        )

    def test_direct_mm_mapping_symmetric(self) -> None:
        cfg = TeleopConfig(invert_yaw=False, invert_pitch=False)
        sample = SimpleNamespace(axis_x=1.0, axis_y=0.5)
        yaw_mm, pitch_mm, target_mm, active = tr._compute_direct_motor_target_mm(
            sample=sample,
            teleop_cfg=cfg,
            motor_limit_mm=8.0,
        )
        self.assertAlmostEqual(yaw_mm, 8.0)
        self.assertAlmostEqual(pitch_mm, 4.0)
        np.testing.assert_allclose(
            target_mm,
            np.asarray([8.0, 4.0, -8.0, -4.0], dtype=np.float64),
            atol=1e-9,
        )
        self.assertEqual(active, (True, False, True, False))

    def test_direct_mm_mapping_with_invert(self) -> None:
        cfg = TeleopConfig(invert_yaw=True, invert_pitch=True)
        sample = SimpleNamespace(axis_x=0.25, axis_y=-0.5)
        yaw_mm, pitch_mm, target_mm, active = tr._compute_direct_motor_target_mm(
            sample=sample,
            teleop_cfg=cfg,
            motor_limit_mm=8.0,
        )
        self.assertAlmostEqual(yaw_mm, -2.0)
        self.assertAlmostEqual(pitch_mm, 4.0)
        np.testing.assert_allclose(
            target_mm,
            np.asarray([-2.0, 4.0, 2.0, -4.0], dtype=np.float64),
            atol=1e-9,
        )
        self.assertEqual(active, (False, False, False, False))

    def test_input_axis_normalization_supports_sdl_int_range(self) -> None:
        self.assertAlmostEqual(PygameGamepadInput._normalize_axis_value(0.5), 0.5)
        self.assertAlmostEqual(PygameGamepadInput._normalize_axis_value(16384), 0.5, places=3)
        self.assertAlmostEqual(PygameGamepadInput._normalize_axis_value(-16384), -0.5, places=3)
        self.assertAlmostEqual(PygameGamepadInput._normalize_axis_value(32767), 1.0, places=4)
        self.assertAlmostEqual(PygameGamepadInput._normalize_axis_value(-32768), -1.0, places=7)

    def test_hold_latch_only_fires_once_per_press(self) -> None:
        latch = HoldLatch()
        self.assertFalse(latch.update(active=True, now_sec=0.0, hold_sec=1.0))
        self.assertFalse(latch.update(active=True, now_sec=0.8, hold_sec=1.0))
        self.assertTrue(latch.update(active=True, now_sec=1.1, hold_sec=1.0))
        self.assertFalse(latch.update(active=True, now_sec=1.6, hold_sec=1.0))
        self.assertFalse(latch.update(active=False, now_sec=1.7, hold_sec=1.0))
        self.assertFalse(latch.update(active=True, now_sec=2.0, hold_sec=1.0))
        self.assertTrue(latch.update(active=True, now_sec=3.1, hold_sec=1.0))

    def test_disconnect_timeout_estop(self) -> None:
        self.assertTrue(
            should_trigger_disconnect_estop(
                now_sec=10.0,
                last_input_ok_sec=9.0,
                timeout_sec=0.5,
                controller_connected=True,
            )
        )
        self.assertTrue(
            should_trigger_disconnect_estop(
                now_sec=10.0,
                last_input_ok_sec=9.9,
                timeout_sec=0.5,
                controller_connected=False,
            )
        )
        self.assertFalse(
            should_trigger_disconnect_estop(
                now_sec=10.0,
                last_input_ok_sec=9.7,
                timeout_sec=0.5,
                controller_connected=True,
            )
        )

    def test_should_send_follow_command_blocked_when_latched(self) -> None:
        self.assertFalse(
            should_send_follow_command(
                command=BridgeCommand.CMD,
                should_send_flag=True,
                estop_latched=True,
            )
        )
        self.assertFalse(
            should_send_follow_command(
                command=BridgeCommand.ESTOP,
                should_send_flag=True,
                estop_latched=False,
            )
        )
        self.assertTrue(
            should_send_follow_command(
                command=BridgeCommand.CMD,
                should_send_flag=True,
                estop_latched=False,
            )
        )

    def test_reset_sequence_order_success(self) -> None:
        calls = []
        tx = FakeTx(calls)
        monitor = FakeMonitor(verify_ok=True)
        bridge = FakeBridge()
        mapper = FakeMapper()

        ok, reason = perform_reset_sequence(
            actuator_tx=tx,
            actuator_monitor=monitor,
            motor_mapper=mapper,
            actuator_ids=(1, 2, 3, 4),
            sim2real_bridge=bridge,
            monitor_settle_sec=0.02,
            dry_run=False,
        )
        self.assertTrue(ok)
        self.assertEqual(reason, "OK")
        self.assertEqual(bridge.reset_called, 1)
        self.assertTrue(monitor.clear_fault_called)
        self.assertEqual(monitor.verify_calls, 1)
        self.assertEqual(calls[0][0], "fault_clear")
        self.assertEqual(calls[1][0], "work_start")
        self.assertEqual(calls[2][0], "follow")
        self.assertEqual(calls[2][3], True)
        self.assertEqual(calls[2][4], "F3_FOLLOW_ZERO")

    def test_reset_sequence_verify_fail(self) -> None:
        calls = []
        tx = FakeTx(calls)
        monitor = FakeMonitor(verify_ok=False)
        bridge = FakeBridge()
        mapper = FakeMapper()

        ok, reason = perform_reset_sequence(
            actuator_tx=tx,
            actuator_monitor=monitor,
            motor_mapper=mapper,
            actuator_ids=(1, 2, 3, 4),
            sim2real_bridge=bridge,
            monitor_settle_sec=0.02,
            dry_run=False,
        )
        self.assertFalse(ok)
        self.assertIn("QUERY_FAIL", reason)
        self.assertEqual(bridge.reset_called, 0)
        self.assertFalse(monitor.clear_fault_called)

    def test_forward_hold_repeats_by_repeat_hz(self) -> None:
        feed_state = self._make_feed_state(
            teleop_enabled=True,
            dry_run=True,
            locked=False,
            repeat_hz=5.0,  # 0.2s period
        )
        ctx = self._make_ctx(feed_state)
        loop_state = tr._create_loop_state(control_hz=30.0)

        with patch(
            "control.teleop_runtime._handle_feed_forward_rising_edge",
            return_value="sent",
        ) as mock_feed_step:
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.00,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.10,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.21,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.42,
            )
        self.assertEqual(mock_feed_step.call_count, 3)

    def test_forward_release_stops_repeat(self) -> None:
        feed_state = self._make_feed_state(
            teleop_enabled=True,
            dry_run=True,
            locked=False,
            repeat_hz=10.0,
        )
        ctx = self._make_ctx(feed_state)
        loop_state = tr._create_loop_state(control_hz=30.0)

        with patch(
            "control.teleop_runtime._handle_feed_forward_rising_edge",
            return_value="sent",
        ) as mock_feed_step:
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.00,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=False),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.05,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=False),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.50,
            )
        self.assertEqual(mock_feed_step.call_count, 1)

    def test_forward_limit_latched_until_release(self) -> None:
        feed_state = self._make_feed_state(
            teleop_enabled=True,
            dry_run=True,
            locked=False,
            repeat_hz=10.0,
        )
        ctx = self._make_ctx(feed_state)
        loop_state = tr._create_loop_state(control_hz=30.0)

        with patch(
            "control.teleop_runtime._handle_feed_forward_rising_edge",
            side_effect=["limit", "sent"],
        ) as mock_feed_step:
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.00,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.20,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=False),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.30,
            )
            tr._update_forward_state_and_feed(
                sample=SimpleNamespace(forward_pressed=True),
                loop_state=loop_state,
                ctx=ctx,
                loop_start=0.40,
            )
        self.assertEqual(mock_feed_step.call_count, 2)

    def test_feed_forward_blocked_when_locked(self) -> None:
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=True, locked=True)
        ctx = self._make_ctx(feed_state)
        with patch("control.teleop_runtime._feed_send_forward_once") as mock_send:
            tr._handle_feed_forward_rising_edge(ctx)
        mock_send.assert_not_called()

    def test_feed_forward_ignored_when_teleop_disabled(self) -> None:
        feed_state = self._make_feed_state(teleop_enabled=False, dry_run=True, locked=False)
        ctx = self._make_ctx(feed_state)
        with patch("control.teleop_runtime._feed_send_forward_once") as mock_send:
            tr._handle_feed_forward_rising_edge(ctx)
        mock_send.assert_not_called()

    def test_feed_forward_blocked_when_estop_latched(self) -> None:
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=True, locked=False)
        ctx = self._make_ctx(feed_state)
        ctx.sim2real_bridge.estop_latched = True
        with patch("control.teleop_runtime._feed_send_forward_once") as mock_send:
            tr._handle_feed_forward_rising_edge(ctx)
        mock_send.assert_not_called()

    def test_feed_forward_reenable_pending_enable_then_send(self) -> None:
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=True, locked=False)
        feed_state.reenable_pending = True
        ctx = self._make_ctx(feed_state)
        with patch("control.teleop_runtime._feed_send_enable", return_value=True) as mock_enable, patch(
            "control.teleop_runtime._feed_send_forward_once",
            return_value=True,
        ) as mock_send, patch("control.teleop_runtime.time.sleep", return_value=None):
            result = tr._handle_feed_forward_rising_edge(ctx)
        self.assertEqual(result, "sent")
        mock_enable.assert_called_once_with(feed_state, state=True)
        mock_send.assert_called_once()
        self.assertFalse(feed_state.reenable_pending)

    def test_estop_latch_disables_and_locks_feed(self) -> None:
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=True, locked=False)
        ctx = self._make_ctx(feed_state)
        with patch("control.teleop_runtime._feed_send_enable", return_value=True) as mock_enable:
            fired = tr._latch_estop(ctx, reason="TEST_ESTOP")
        self.assertTrue(fired)
        self.assertTrue(ctx.sim2real_bridge.estop_latched)
        self.assertTrue(feed_state.locked)
        mock_enable.assert_called_once_with(feed_state, state=False)

    def test_reset_sequence_success_includes_feed_disable_enable(self) -> None:
        calls = []
        tx = FakeTx(calls)
        monitor = FakeMonitor(verify_ok=True)
        bridge = FakeBridge()
        mapper = FakeMapper()
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=False, locked=True)

        with patch("control.teleop_runtime._feed_send_enable", side_effect=[True, True]) as mock_enable:
            ok, reason = perform_reset_sequence(
                actuator_tx=tx,
                actuator_monitor=monitor,
                motor_mapper=mapper,
                actuator_ids=(1, 2, 3, 4),
                sim2real_bridge=bridge,
                monitor_settle_sec=0.02,
                dry_run=False,
                feed_state=feed_state,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "OK")
        self.assertFalse(feed_state.locked)
        self.assertEqual(mock_enable.call_count, 2)

    def test_reset_sequence_fail_when_feed_reenable_fails(self) -> None:
        calls = []
        tx = FakeTx(calls)
        monitor = FakeMonitor(verify_ok=True)
        bridge = FakeBridge()
        mapper = FakeMapper()
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=False, locked=True)

        with patch("control.teleop_runtime._feed_send_enable", side_effect=[True, False]) as mock_enable:
            ok, reason = perform_reset_sequence(
                actuator_tx=tx,
                actuator_monitor=monitor,
                motor_mapper=mapper,
                actuator_ids=(1, 2, 3, 4),
                sim2real_bridge=bridge,
                monitor_settle_sec=0.02,
                dry_run=False,
                feed_state=feed_state,
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "FEED_ENABLE_FAIL")
        self.assertTrue(feed_state.locked)
        self.assertEqual(mock_enable.call_count, 2)

    def test_reset_sequence_feed_only_success(self) -> None:
        bridge = FakeBridge()
        mapper = FakeMapper()
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=False, locked=True)

        with patch("control.teleop_runtime._feed_send_enable", side_effect=[True, True]) as mock_enable:
            ok, reason = perform_reset_sequence(
                actuator_tx=None,
                actuator_monitor=None,
                motor_mapper=mapper,
                actuator_ids=(1, 2, 3, 4),
                sim2real_bridge=bridge,
                monitor_settle_sec=0.02,
                dry_run=False,
                feed_state=feed_state,
                allow_feed_without_actuator=True,
            )
        self.assertTrue(ok)
        self.assertEqual(reason, "OK_FEED_ONLY")
        self.assertFalse(feed_state.locked)
        self.assertEqual(bridge.reset_called, 1)
        self.assertEqual(mock_enable.call_count, 2)

    def test_reset_sequence_feed_only_fail_keeps_latch(self) -> None:
        bridge = FakeBridge()
        mapper = FakeMapper()
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=False, locked=True)

        with patch("control.teleop_runtime._feed_send_enable", side_effect=[True, False]) as mock_enable:
            ok, reason = perform_reset_sequence(
                actuator_tx=None,
                actuator_monitor=None,
                motor_mapper=mapper,
                actuator_ids=(1, 2, 3, 4),
                sim2real_bridge=bridge,
                monitor_settle_sec=0.02,
                dry_run=False,
                feed_state=feed_state,
                allow_feed_without_actuator=True,
            )
        self.assertFalse(ok)
        self.assertEqual(reason, "FEED_ENABLE_FAIL")
        self.assertTrue(feed_state.locked)
        self.assertEqual(bridge.reset_called, 0)
        self.assertEqual(mock_enable.call_count, 2)

    def test_port_conflict_validation(self) -> None:
        cfg = FeedConfig(port="/dev/ttyUSB0", teleop_enabled=True)
        cfg.validate()
        with self.assertRaises(ValueError):
            tr._validate_feed_port_conflict(
                dry_run=False,
                actuator_serial_port="/dev/ttyUSB0",
                feed_cfg=cfg,
            )
        tr._validate_feed_port_conflict(
            dry_run=True,
            actuator_serial_port="/dev/ttyUSB0",
            feed_cfg=cfg,
        )

    def test_run_teleop_feed_only_mode_skips_actuator_and_bridge(self) -> None:
        runtime_cfg = self._make_runtime_cfg()
        teleop_cfg = TeleopConfig(allow_feed_without_actuator=True)
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=True, locked=False)

        with patch(
            "control.teleop_runtime.load_dagger_sim2real_runtime_config",
            return_value=runtime_cfg,
        ), patch(
            "control.teleop_runtime.load_teleop_config",
            return_value=teleop_cfg,
        ), patch(
            "control.teleop_runtime.PygameGamepadInput",
            FakeRuntimeGamepad,
        ), patch(
            "control.teleop_runtime.Sim2RealBridge",
            FakeRuntimeBridge,
        ), patch(
            "control.teleop_runtime.MotorMapper",
            FakeMapper,
        ), patch(
            "control.teleop_runtime._setup_actuator_io"
        ) as mock_setup_actuator, patch(
            "control.teleop_runtime._setup_feed_runtime",
            return_value=feed_state,
        ) as mock_setup_feed, patch(
            "control.teleop_runtime._run_boot_reset"
        ) as mock_boot_reset, patch(
            "control.teleop_runtime._compute_direct_motor_target_mm"
        ) as mock_compute_target, patch(
            "control.teleop_runtime._send_direct_target_and_check_limits"
        ) as mock_bridge_send, patch(
            "control.teleop_runtime._sleep_for_rate",
            return_value=None,
        ):
            rc = tr.run_teleop(
                config_path="control/sim2real_config.yaml",
                dry_run=True,
                debug_input=False,
                list_controllers_only=False,
            )

        self.assertEqual(rc, 0)
        mock_setup_actuator.assert_not_called()
        mock_boot_reset.assert_not_called()
        mock_compute_target.assert_not_called()
        mock_bridge_send.assert_not_called()
        self.assertEqual(mock_setup_feed.call_count, 1)
        self.assertFalse(mock_setup_feed.call_args.kwargs["check_port_conflict"])

    def test_run_teleop_default_mode_keeps_actuator_path(self) -> None:
        runtime_cfg = self._make_runtime_cfg()
        teleop_cfg = TeleopConfig(allow_feed_without_actuator=False)
        feed_state = self._make_feed_state(teleop_enabled=True, dry_run=True, locked=False)

        with patch(
            "control.teleop_runtime.load_dagger_sim2real_runtime_config",
            return_value=runtime_cfg,
        ), patch(
            "control.teleop_runtime.load_teleop_config",
            return_value=teleop_cfg,
        ), patch(
            "control.teleop_runtime.PygameGamepadInput",
            FakeRuntimeGamepad,
        ), patch(
            "control.teleop_runtime.Sim2RealBridge",
            FakeRuntimeBridge,
        ), patch(
            "control.teleop_runtime.MotorMapper",
            FakeMapper,
        ), patch(
            "control.teleop_runtime._setup_actuator_io",
            return_value=(None, None, None),
        ) as mock_setup_actuator, patch(
            "control.teleop_runtime._setup_feed_runtime",
            return_value=feed_state,
        ) as mock_setup_feed, patch(
            "control.teleop_runtime._run_boot_reset"
        ) as mock_boot_reset, patch(
            "control.teleop_runtime._compute_direct_motor_target_mm",
            return_value=(
                0.1,
                0.1,
                np.zeros(4, dtype=np.float64),
                (False, False, False, False),
            ),
        ) as mock_compute_target, patch(
            "control.teleop_runtime._send_direct_target_and_check_limits"
        ) as mock_bridge_send, patch(
            "control.teleop_runtime._sleep_for_rate",
            return_value=None,
        ):
            rc = tr.run_teleop(
                config_path="control/sim2real_config.yaml",
                dry_run=True,
                debug_input=False,
                list_controllers_only=False,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(mock_setup_actuator.call_count, 1)
        self.assertEqual(mock_boot_reset.call_count, 1)
        self.assertGreaterEqual(mock_compute_target.call_count, 1)
        self.assertGreaterEqual(mock_bridge_send.call_count, 1)
        self.assertEqual(mock_setup_feed.call_count, 1)
        self.assertTrue(mock_setup_feed.call_args.kwargs["check_port_conflict"])


class TeleopConfigCompatTests(unittest.TestCase):
    def _write_yaml(self, yaml_text: str) -> str:
        fd, path = tempfile.mkstemp(prefix="teleop_cfg_", suffix=".yaml")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(yaml_text))
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_load_teleop_config_public_fields(self) -> None:
        path = self._write_yaml(
            """
            sim2real:
              teleop:
                controller_index: 2
                invert_yaw: true
                invert_pitch: false
                hold_sec: 0.7
                disconnect_timeout_sec: 0.6
            """
        )
        cfg = load_teleop_config(path)
        self.assertEqual(cfg.controller_index, 2)
        self.assertTrue(cfg.invert_yaw)
        self.assertFalse(cfg.invert_pitch)
        self.assertAlmostEqual(cfg.hold_sec, 0.7)
        self.assertAlmostEqual(cfg.estop_hold_sec, 0.7)
        self.assertAlmostEqual(cfg.reset_hold_sec, 0.7)
        self.assertAlmostEqual(cfg.disconnect_timeout_sec, 0.6)
        self.assertEqual(cfg.estop_button, "south")
        self.assertEqual(cfg.reset_combo, ("west", "west"))
        self.assertFalse(cfg.allow_feed_without_actuator)
        self.assertEqual(cfg.deprecation_warnings, ())

    def test_load_teleop_config_removed_rate_fields_raise(self) -> None:
        path = self._write_yaml(
            """
            sim2real:
              teleop:
                deadzone: 0.2
                max_yaw_rate_rad_s: 1.1
            """
        )
        with self.assertRaises(ValueError) as ctx:
            load_teleop_config(path)
        self.assertIn("deadzone", str(ctx.exception))
        self.assertIn("max_yaw_rate_rad_s", str(ctx.exception))

    def test_load_teleop_config_allow_feed_without_actuator_true(self) -> None:
        path = self._write_yaml(
            """
            sim2real:
              teleop:
                allow_feed_without_actuator: true
            """
        )
        cfg = load_teleop_config(path)
        self.assertTrue(cfg.allow_feed_without_actuator)

    def test_load_teleop_config_legacy_hold_overrides_new_hold(self) -> None:
        path = self._write_yaml(
            """
            sim2real:
              teleop:
                hold_sec: 0.9
                estop_hold_sec: 1.2
                reset_hold_sec: 1.3
            """
        )
        cfg = load_teleop_config(path)
        self.assertAlmostEqual(cfg.hold_sec, 0.9)
        self.assertAlmostEqual(cfg.estop_hold_sec, 1.2)
        self.assertAlmostEqual(cfg.reset_hold_sec, 1.3)
        self.assertTrue(any("estop_hold_sec" in msg for msg in cfg.deprecation_warnings))
        self.assertTrue(any("reset_hold_sec" in msg for msg in cfg.deprecation_warnings))

    def test_load_teleop_config_legacy_input_fields_still_work(self) -> None:
        path = self._write_yaml(
            """
            sim2real:
              teleop:
                controller_name_contains: "GamepadX"
                left_stick_x_axis: 3
                left_stick_y_axis: 4
                forward_button: "south"
                estop_button: "start"
                reset_combo: ["start", "back"]
            """
        )
        cfg = load_teleop_config(path)
        self.assertEqual(cfg.controller_name_contains, "GamepadX")
        self.assertEqual(cfg.left_stick_x_axis, 3)
        self.assertEqual(cfg.left_stick_y_axis, 4)
        self.assertEqual(cfg.forward_button, "south")
        self.assertEqual(cfg.estop_button, "start")
        self.assertEqual(cfg.reset_combo, ("start", "back"))
        self.assertGreaterEqual(len(cfg.deprecation_warnings), 6)

    def test_parser_public_surface_hides_legacy_args(self) -> None:
        parser = build_arg_parser()
        help_text = parser.format_help()
        self.assertIn("--dry-run", help_text)
        self.assertIn("--debug-input", help_text)
        self.assertIn("--list-controllers", help_text)
        self.assertNotIn("--config", help_text)
        self.assertNotIn("--controller-index", help_text)
        self.assertNotIn("--debug-input-hz", help_text)

    def test_parser_rejects_removed_debug_hz_arg(self) -> None:
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--debug-input-hz", "8"])

    def test_parser_accepts_list_controllers_arg(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--list-controllers"])
        self.assertTrue(args.list_controllers)

    def test_parser_rejects_removed_controller_index_arg(self) -> None:
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--controller-index", "0"])

    def test_parser_rejects_removed_config_arg(self) -> None:
        parser = build_arg_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--config", "control/sim2real_config.yaml"])


if __name__ == "__main__":
    unittest.main()
