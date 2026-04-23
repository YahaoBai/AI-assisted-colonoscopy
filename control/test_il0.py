import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import control.il0 as il0


class IL0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="il0_test_"))
        self.addCleanup(lambda: shutil.rmtree(self.tmpdir, ignore_errors=True))

    def test_toggle_auto_recording_starts_session_and_enables_autopilot(self) -> None:
        runtime = il0.RecordingRuntime()

        runtime, autopilot_on, message = il0.toggle_auto_recording(
            runtime,
            autopilot_on=False,
            base_output_root=self.tmpdir,
        )

        self.assertTrue(autopilot_on)
        self.assertEqual(runtime.active_mode, "auto")
        self.assertIsNotNone(runtime.recorder)
        assert runtime.recorder is not None
        self.assertTrue(runtime.recorder.csv_path.is_file())
        self.assertIn("自动录制已开启", message)
        il0.close_active_session(runtime)

    def test_toggle_auto_recording_second_time_stops_session_and_disables_autopilot(self) -> None:
        runtime = il0.RecordingRuntime()
        runtime, autopilot_on, _ = il0.toggle_auto_recording(
            runtime,
            autopilot_on=False,
            base_output_root=self.tmpdir,
        )

        runtime, autopilot_on, message = il0.toggle_auto_recording(
            runtime,
            autopilot_on=autopilot_on,
            base_output_root=self.tmpdir,
        )

        self.assertFalse(autopilot_on)
        self.assertIsNone(runtime.active_mode)
        self.assertIsNone(runtime.recorder)
        self.assertIn("自动录制已关闭", message)

    def test_toggle_manual_recording_starts_and_stops_in_manual_mode(self) -> None:
        runtime = il0.RecordingRuntime()

        runtime, autopilot_on, start_message = il0.toggle_manual_recording(
            runtime,
            autopilot_on=False,
            base_output_root=self.tmpdir,
        )
        self.assertFalse(autopilot_on)
        self.assertEqual(runtime.active_mode, "manual")
        self.assertIsNotNone(runtime.recorder)
        self.assertIn("手动录制已开启", start_message)

        runtime, autopilot_on, stop_message = il0.toggle_manual_recording(
            runtime,
            autopilot_on=False,
            base_output_root=self.tmpdir,
        )
        self.assertFalse(autopilot_on)
        self.assertIsNone(runtime.active_mode)
        self.assertIsNone(runtime.recorder)
        self.assertIn("手动录制已关闭", stop_message)

    def test_toggle_manual_recording_is_ignored_in_auto_mode(self) -> None:
        runtime = il0.RecordingRuntime()
        runtime, autopilot_on, _ = il0.toggle_auto_recording(
            runtime,
            autopilot_on=False,
            base_output_root=self.tmpdir,
        )
        auto_root = runtime.recorder.output_root if runtime.recorder is not None else None

        runtime, autopilot_on, message = il0.toggle_manual_recording(
            runtime,
            autopilot_on=autopilot_on,
            base_output_root=self.tmpdir,
        )

        self.assertTrue(autopilot_on)
        self.assertEqual(runtime.active_mode, "auto")
        self.assertEqual(runtime.recorder.output_root if runtime.recorder is not None else None, auto_root)
        self.assertIn("忽略手动录制请求", message)
        il0.close_active_session(runtime)

    def test_toggling_auto_closes_manual_session_first(self) -> None:
        runtime = il0.RecordingRuntime()
        runtime, autopilot_on, _ = il0.toggle_manual_recording(
            runtime,
            autopilot_on=False,
            base_output_root=self.tmpdir,
        )
        old_manual_root = runtime.recorder.output_root if runtime.recorder is not None else None

        runtime, autopilot_on, message = il0.toggle_auto_recording(
            runtime,
            autopilot_on=False,
            base_output_root=self.tmpdir,
        )

        self.assertTrue(autopilot_on)
        self.assertEqual(runtime.active_mode, "auto")
        self.assertIsNotNone(runtime.recorder)
        self.assertNotEqual(runtime.recorder.output_root, old_manual_root)
        self.assertIn("自动录制已开启", message)
        il0.close_active_session(runtime)

    def test_session_recorder_creates_raw_mask_overlay_and_csv(self) -> None:
        recorder = il0.SessionRecorder.create(self.tmpdir, mode="manual")
        self.addCleanup(recorder.close)

        self.assertTrue(recorder.raw_dir.is_dir())
        self.assertTrue(recorder.mask_dir.is_dir())
        self.assertTrue(recorder.overlay_dir.is_dir())
        self.assertTrue(recorder.csv_path.is_file())

    def test_write_metric_row_records_expected_fields(self) -> None:
        recorder = il0.SessionRecorder.create(self.tmpdir, mode="auto")
        self.addCleanup(recorder.close)

        il0.write_metric_row(
            recorder,
            frame_idx=3,
            timestamp_sec=12.34,
            frame_shape=(256, 256, 3),
            scope_center=(128, 128),
            lumen_center=(130, 140),
            error_x_px=2,
            error_y_px=12,
            error_norm_px=float(np.hypot(2, 12)),
            inference_ms=5.5,
            status_text="OK",
            record_mode="auto",
            autopilot_on=True,
            policy_step_yaw=0.01,
            policy_step_pitch=-0.02,
        )
        recorder.close()

        with recorder.csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["record_mode"], "auto")
        self.assertEqual(row["autopilot_on"], "1")
        self.assertEqual(row["policy_step_yaw"], "0.01000000")
        self.assertEqual(row["policy_step_pitch"], "-0.02000000")


if __name__ == "__main__":
    unittest.main()
