"""Generate synthetic fixtures; this is not an astronomical observation."""
from pathlib import Path
import cv2
import numpy as np
p = Path("sample_data")
p.mkdir(exist_ok=True)
im = np.zeros((768, 768, 3), dtype=np.uint8)
cv2.circle(im, (384, 384), 320, (210, 210, 210), -1)
for x,y in [(280,310),(295,320),(475,425)]:
    cv2.circle(im, (x,y), 7, (35,35,35), -1)
cv2.imwrite(str(p / "synthetic_sun.png"), im)
cv2.imwrite(str(p / "synthetic_blank.png"), np.zeros_like(im))
