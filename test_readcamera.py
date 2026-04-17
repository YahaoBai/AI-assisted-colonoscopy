import math
import unittest

import numpy as np

import readcamera


class ReadCameraHelperTests(unittest.TestCase):
    def test_compute_center_error_with_lumen_center(self) -> None:
        scope_center, error_x_px, error_y_px, error_norm_px = readcamera.compute_center_error(
            lumen_center=(400, 200),
            frame_shape=(480, 640, 3),
        )
        self.assertEqual(scope_center, (320, 240))
        self.assertEqual(error_x_px, 80)
        self.assertEqual(error_y_px, -40)
        self.assertTrue(math.isclose(float(error_norm_px), math.hypot(80, -40)))

    def test_compute_center_error_without_lumen_center(self) -> None:
        scope_center, error_x_px, error_y_px, error_norm_px = readcamera.compute_center_error(
            lumen_center=None,
            frame_shape=(480, 640, 3),
        )
        self.assertEqual(scope_center, (320, 240))
        self.assertIsNone(error_x_px)
        self.assertIsNone(error_y_px)
        self.assertIsNone(error_norm_px)

    def test_mask_to_u8_returns_binary_mask(self) -> None:
        mask_u8 = readcamera.mask_to_u8(
            mask=np.array([[0, 1], [1, 0]], dtype=np.uint8),
            frame_shape=(2, 2),
        )
        np.testing.assert_array_equal(
            mask_u8,
            np.array([[0, 255], [255, 0]], dtype=np.uint8),
        )


if __name__ == "__main__":
    unittest.main()
