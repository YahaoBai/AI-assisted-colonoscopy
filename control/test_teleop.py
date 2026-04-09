import os
import tempfile
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np

from control.sim2real_bridge import BridgeCommand, FaultClearVerification
from control.teleop import (
    HoldLatch,
    PygameGamepadInput,
    apply_deadzone,
    axis_to_rate,
    build_arg_parser,
    load_teleop_config,
    perform_reset_sequence,
    should_send_follow_command,
    should_trigger_disconnect_estop,
)


class FakeMapper:
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

    def reset(self):
        self.reset_called += 1


class TeleopCoreTests(unittest.TestCase):
    def test_apply_deadzone_and_rescale(self) -> None:
        self.assertAlmostEqual(apply_deadzone(0.10, 0.15), 0.0)
        self.assertAlmostEqual(apply_deadzone(-0.10, 0.15), 0.0)
        # (0.30 - 0.15) / (1 - 0.15) = 0.17647...
        self.assertAlmostEqual(apply_deadzone(0.30, 0.15), 0.1764705882, places=7)

    def test_axis_to_rate_with_invert(self) -> None:
        rate = axis_to_rate(raw_axis=0.50, deadzone=0.10, max_rate_rad_s=0.9, invert=False)
        self.assertGreater(rate, 0.0)

        rate_inv = axis_to_rate(raw_axis=0.50, deadzone=0.10, max_rate_rad_s=0.9, invert=True)
        self.assertAlmostEqual(rate_inv, -rate, places=7)

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


class TeleopConfigCompatTests(unittest.TestCase):
    def _write_yaml(self, yaml_text: str) -> str:
        fd, path = tempfile.mkstemp(prefix="teleop_cfg_", suffix=".yaml")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(yaml_text))
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_load_teleop_config_public_8_fields(self) -> None:
        path = self._write_yaml(
            """
            sim2real:
              teleop:
                controller_index: 2
                deadzone: 0.2
                max_yaw_rate_rad_s: 1.1
                max_pitch_rate_rad_s: 1.2
                invert_yaw: true
                invert_pitch: false
                hold_sec: 0.7
                disconnect_timeout_sec: 0.6
            """
        )
        cfg = load_teleop_config(path)
        self.assertEqual(cfg.controller_index, 2)
        self.assertAlmostEqual(cfg.deadzone, 0.2)
        self.assertAlmostEqual(cfg.max_yaw_rate_rad_s, 1.1)
        self.assertAlmostEqual(cfg.max_pitch_rate_rad_s, 1.2)
        self.assertTrue(cfg.invert_yaw)
        self.assertFalse(cfg.invert_pitch)
        self.assertAlmostEqual(cfg.hold_sec, 0.7)
        self.assertAlmostEqual(cfg.estop_hold_sec, 0.7)
        self.assertAlmostEqual(cfg.reset_hold_sec, 0.7)
        self.assertAlmostEqual(cfg.disconnect_timeout_sec, 0.6)
        self.assertEqual(cfg.estop_button, "south")
        self.assertEqual(cfg.reset_combo, ("west", "west"))
        self.assertEqual(cfg.deprecation_warnings, ())

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
