import os
import tempfile
import textwrap
import unittest

from control.feed import (
    CHK,
    FeedConfig,
    build_arg_parser,
    build_en_control_frame,
    build_pos_control_frame,
    compute_next_forward_target,
    resolve_feed_config,
    resolve_forward_dir,
)


class FeedProtocolTests(unittest.TestCase):
    def test_build_en_control_frame_enable_disable(self) -> None:
        enable = build_en_control_frame(addr=1, state=True, sync_start=False)
        disable = build_en_control_frame(addr=1, state=False, sync_start=False)
        self.assertEqual(enable, bytes([0x01, 0xF3, 0xAB, 0x01, 0x00, CHK]))
        self.assertEqual(disable, bytes([0x01, 0xF3, 0xAB, 0x00, 0x00, CHK]))

    def test_build_pos_control_frame_matches_reference_shape(self) -> None:
        # 3200 = 0x00000C80
        frame = build_pos_control_frame(
            addr=1,
            dir_flag=0,
            vel=100,
            acc=0,
            clk=3200,
            relative_mode=True,
            sync_start=False,
        )
        expected = bytes(
            [
                0x01,
                0xFD,
                0x00,
                0x00,
                0x64,
                0x00,
                0x00,
                0x00,
                0x0C,
                0x80,
                0x00,
                0x00,
                CHK,
            ]
        )
        self.assertEqual(frame, expected)

    def test_resolve_forward_dir(self) -> None:
        self.assertEqual(resolve_forward_dir(False), 0)
        self.assertEqual(resolve_forward_dir(True), 1)

    def test_compute_next_forward_target_limit_reject(self) -> None:
        ok, next_p = compute_next_forward_target(
            current_pulses=900,
            step_pulses=200,
            min_pulses=0,
            max_pulses=1000,
        )
        self.assertFalse(ok)
        self.assertEqual(next_p, 900)

        ok2, next_p2 = compute_next_forward_target(
            current_pulses=700,
            step_pulses=200,
            min_pulses=0,
            max_pulses=1000,
        )
        self.assertTrue(ok2)
        self.assertEqual(next_p2, 900)


class FeedConfigTests(unittest.TestCase):
    def _write_yaml(self, yaml_text: str) -> str:
        fd, path = tempfile.mkstemp(prefix="feed_cfg_", suffix=".yaml")
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(yaml_text))
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))
        return path

    def test_resolve_feed_config_yaml_only_with_dry_run(self) -> None:
        path = self._write_yaml(
            """
            sim2real:
              feed:
                port: "/dev/ttyUSB9"
                baudrate: 9600
                timeout: 0.2
                addr: 7
                microstep: 32
                steps_per_rev: 200
                step_pulses: 150
                repeat_hz: 8.5
                default_vel: 120
                default_acc: 8
                invert_dir: true
                min_pulses: 0
                max_pulses: 5000
                enable_on_start: true
                disable_on_exit: true
                teleop_enabled: true
            """
        )

        cfg = resolve_feed_config(dry_run=True, config_path=path)
        self.assertEqual(cfg.port, "/dev/ttyUSB9")
        self.assertEqual(cfg.baudrate, 9600)
        self.assertEqual(cfg.timeout, 0.2)
        self.assertEqual(cfg.addr, 7)
        self.assertEqual(cfg.step_pulses, 150)
        self.assertAlmostEqual(cfg.repeat_hz, 8.5)
        self.assertEqual(cfg.default_vel, 120)
        self.assertEqual(cfg.default_acc, 8)
        self.assertTrue(cfg.invert_dir)
        self.assertEqual(cfg.min_pulses, 0)
        self.assertEqual(cfg.max_pulses, 5000)
        self.assertTrue(cfg.teleop_enabled)
        self.assertTrue(cfg.dry_run)

    def test_feed_config_rejects_invalid_repeat_hz(self) -> None:
        with self.assertRaises(ValueError):
            FeedConfig(repeat_hz=0.0).validate()
        with self.assertRaises(ValueError):
            FeedConfig(repeat_hz=101.0).validate()


class FeedCliTests(unittest.TestCase):
    def test_parser_supports_minimal_cli_surface(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(["--list-ports", "--dry-run"])
        self.assertTrue(args.list_ports)
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
