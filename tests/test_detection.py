import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from solar_insight.cli import analyze
from solar_insight.detection import group_sunspots_with_labels

class DetectionTests(unittest.TestCase):
    def test_synthetic_image_and_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            image = np.zeros((768,768,3), dtype=np.uint8)
            cv2.circle(image,(384,384),320,(210,210,210),-1)
            for xy in [(280,310),(295,320),(475,425)]:
                cv2.circle(image,xy,7,(35,35,35),-1)
            cv2.imwrite(str(p/'sun.png'),image)
            out = analyze(p/'sun.png',p/'out')
            self.assertEqual(out['spot_count'],3)
            self.assertEqual(out['group_count'],2)
            self.assertEqual(out['relative_number'],23)
            self.assertTrue((p/'out/sun_comparison.png').exists())
    def test_blank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'blank.png'
            cv2.imwrite(str(p),np.zeros((100,100,3),dtype=np.uint8))
            with self.assertRaises(ValueError):analyze(p,Path(d)/'out')
    def test_legacy_seed_grouping_is_not_transitive(self):
        self.assertEqual(len(group_sunspots_with_labels([(0,0),(70,0),(140,0)])),2)

if __name__ == '__main__':unittest.main()
