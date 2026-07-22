import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import control.teleop2 as t2
import control.teleop_runtime as tr
from control.sim2real_bridge import BridgeCommand


class FakeMapper:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs

    def mm_targets_to_counts(self, mm_targets: np.ndarray) -> np.ndarray:
        arr = np.asarray(mm_targets, dtype=np.float64).reshape(-1)
        if arr.shape != (4,):
            raise ValueError("shape mismatch")
        return np.asarray([1000, 1001, 1002, 1003], dtype=np.int32)


class FakeRuntimeBridge:
    def __init__(self, *args, **kwargs):
        _ = args
        _ = kwargs
        self.estop_latched = False
        self.motor_target_mm = np.zeros(4, dtype=np.float64)
        self.motor_limit_mm = 8.0

    def reset(self) -> None:
        return None

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
    samples = []

    def __init__(self, cfg):
        _ = cfg
        self.controller_index = 0
        self.controller_name = "fake-gamepad"
        self._poll_index = 0

    def list_controllers(self):
        return [(0, "fake-gamepad", True)]

    def open(self) -> None:
        return None

    def poll(self):
        if self._poll_index >= len(self.samples):
            raise KeyboardInterrupt()
        sample = self.samples[self._poll_index]
        self._poll_index += 1
        return sample

    def close(self) -> None:
        return None


class FakeTogglePoller:
    def __init__(self, toggles):
        self._toggles = list(toggles)
        self.reason = "OK"
        self.opened = False
        self.closed = False

    def open(self) -> bool:
        self.opened = True
        return True

    def poll_toggle(self) -> bool:
        if not self._toggles:
            return False
        return bool(self._toggles.pop(0))

    def close(self) -> None:
        self.closed = True


class Teleop2Tests(unittest.TestCase):
    def test_toggle_recording_session_creates_unique_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            capture_root = Path(tmp_dir)
            with patch("control.teleop2.TELEOP_CAPTURE_ROOT", capture_root), patch(
                "control.teleop2.time.strftime",
                return_value="20260424_180000",
            ), patch("control.teleop2.log_event", return_value=None):
                recorder = t2._toggle_recording_session(None, now_sec=0.0)
                self.assertIsNotNone(recorder)
                assert recorder is not None
                first_dir = recorder.output_dir

                recorder = t2._toggle_recording_session(recorder, now_sec=0.0)
                self.assertIsNone(recorder)

                recorder = t2._toggle_recording_session(None, now_sec=0.0)
                self.assertIsNotNone(recorder)
                assert recorder is not None
                second_dir = recorder.output_dir
                recorder.close()

            self.assertNotEqual(first_dir, second_dir)
            self.assertTrue(first_dir.exists())
            self.assertTrue(second_dir.exists())
            self.assertEqual(first_dir.name, "20260424_180000")
            self.assertEqual(second_dir.name, "20260424_180000_01")

    def test_run_teleop2_records_only_while_toggled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            capture_root = Path(tmp_dir)
            feed_state = tr.FeedRuntimeState(
                enabled=False,
                cfg=None,
                serial_link=None,
                current_pulses=0,
                locked=False,
            )
            FakeRuntimeGamepad.samples = [
                SimpleNamespace(
                    timestamp_sec=0.0,
                    axis_x=0.7,
                    axis_y=-0.7,
                    forward_pressed=False,
                    backward_pressed=False,
                    hold_pressed=False,
                    connected=True,
                ),
                SimpleNamespace(
                    timestamp_sec=0.1,
                    axis_x=0.7,
                    axis_y=-0.7,
                    forward_pressed=False,
                    backward_pressed=False,
                    hold_pressed=False,
                    connected=True,
                ),
                SimpleNamespace(
                    timestamp_sec=0.2,
                    axis_x=0.7,
                    axis_y=-0.7,
                    forward_pressed=False,
                    backward_pressed=False,
                    hold_pressed=False,
                    connected=True,
                ),
                SimpleNamespace(
                    timestamp_sec=0.3,
                    axis_x=0.7,
                    axis_y=-0.7,
                    forward_pressed=False,
                    backward_pressed=False,
                    hold_pressed=False,
                    connected=True,
                ),
            ]

            with patch("control.teleop2.PygameGamepadInput", FakeRuntimeGamepad), patch(
                "control.teleop2.Sim2RealBridge",
                FakeRuntimeBridge,
            ), patch(
                "control.teleop2.MotorMapper",
                FakeMapper,
            ), patch(
                "control.teleop2._setup_actuator_io",
                return_value=(None, None, None),
            ), patch(
                "control.teleop2._setup_feed_runtime",
                return_value=feed_state,
            ), patch(
                "control.teleop2._run_boot_reset",
                return_value=None,
            ), patch(
                "control.teleop2._sleep_for_rate",
                return_value=None,
            ), patch(
                "control.teleop2.TELEOP_CAPTURE_ROOT",
                capture_root,
            ), patch(
                "control.teleop2.time.strftime",
                return_value="20260424_180500",
            ):
                rc = t2.run_teleop2(
                    config_path="control/sim2real_config.yaml",
                    dry_run=True,
                    debug_input=False,
                    list_controllers_only=False,
                    toggle_poller_factory=lambda: FakeTogglePoller([True, False, True, True]),
                )

            self.assertEqual(rc, 0)
            csv_files = sorted(capture_root.glob("*/controls.csv"))
            self.assertEqual(len(csv_files), 2)

            with csv_files[0].open("r", encoding="utf-8", newline="") as f:
                rows_first = list(csv.DictReader(f))
            with csv_files[1].open("r", encoding="utf-8", newline="") as f:
                rows_second = list(csv.DictReader(f))

            self.assertEqual(len(rows_first), 2)
            self.assertEqual(len(rows_second), 1)

            self.assertAlmostEqual(float(rows_first[0]["m1_target_mm"]), 5.6, places=6)
            self.assertAlmostEqual(float(rows_first[0]["m2_target_mm"]), 5.6, places=6)
            self.assertEqual(int(rows_first[0]["feed_delta_pulses"]), 0)
            self.assertEqual(int(rows_first[0]["estop_latched"]), 0)


if __name__ == "__main__":
    unittest.main()
