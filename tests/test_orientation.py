import math
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from solar_insight.cli import analyze
from solar_insight.orientation import axis_vectors, parse_observed_at, solar_orientation


class OrientationTests(unittest.TestCase):
    def test_timezone_conversion(self):
        self.assertEqual(parse_observed_at("2025-04-30T14:08:10"),
                         parse_observed_at("2025-04-30T05:08:10Z"))
        self.assertEqual(parse_observed_at("2025-04-30T14:08:10+09:00"),
                         parse_observed_at("2025-04-30T05:08:10Z"))
        with self.assertRaises(ValueError):
            parse_observed_at("2025-04-30T14:08:10", "invalid/zone")

    def test_cardinal_signs_and_mirror(self):
        v = axis_vectors(0)
        self.assertEqual(v["N"], (0, -1))
        self.assertEqual(v["W"], (1, 0))
        self.assertEqual(v["E"], (-1, 0))
        self.assertAlmostEqual(axis_vectors(90)["N"][0], -1)
        for a in [-140, -20, 0, 90, 170]:
            v, m = axis_vectors(a), axis_vectors(a, True)
            self.assertAlmostEqual(sum(x*y for x, y in zip(v["N"], v["W"])), 0)
            for label in v:
                self.assertAlmostEqual(math.hypot(*v[label]), 1)
                self.assertEqual(m[label], (-v[label][0], v[label][1]))

    def test_partial_metadata_rejected(self):
        with self.assertRaises(ValueError):
            analyze("unused.png", "unused", latitude=35)

    def test_actual_ephemeris_and_site_dependence(self):
        try:
            import sunpy
        except ImportError:
            self.skipTest("Optional solar dependencies not installed")
        a = solar_orientation("2025-04-30T14:08:10", 35, 137)
        b = solar_orientation("2025-04-30T05:08:10Z", 35, 137)
        self.assertAlmostEqual(a["solar_north_from_zenith_deg"],
                               b["solar_north_from_zenith_deg"], places=9)
        c = solar_orientation("2025-04-30T14:08:10", 40, 145)
        self.assertNotAlmostEqual(a["solar_north_from_zenith_deg"],
                                  c["solar_north_from_zenith_deg"], places=2)
        self.assertAlmostEqual(a["p_angle_deg"], c["p_angle_deg"])
        self.assertLess(abs(a["p_angle_deg"]), 27)
        with self.assertRaises(ValueError):
            solar_orientation("2025-04-30T00:00:00", 35, 137)

    def test_overlay_does_not_change_counts(self):
        try:
            import sunpy
        except ImportError:
            self.skipTest("Optional solar dependencies not installed")
        image = np.zeros((768, 768, 3), np.uint8)
        cv2.circle(image, (384, 384), 320, (210, 210, 210), -1)
        for xy in [(280, 310), (295, 320), (475, 425)]:
            cv2.circle(image, xy, 7, (35, 35, 35), -1)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            cv2.imwrite(str(p/"sun.png"), image)
            a = analyze(p/"sun.png", p/"plain")
            b = analyze(p/"sun.png", p/"axes", observed_at="2025-04-30T14:08:10",
                        latitude=35, longitude=137)
            for key in ("spot_count", "group_count", "relative_number"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(b["relative_number"], 23)
            self.assertTrue((p/"axes/sun_annotated.png").exists())
