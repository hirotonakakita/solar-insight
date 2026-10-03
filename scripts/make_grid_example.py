"""Recreate the README grid sample with approximate Yokkaichi coordinates."""
import json
import tempfile
from pathlib import Path
import cv2
from solar_insight.cli import analyze

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as folder:
    result = analyze(root / "sample_data/20250430140810.png", folder,
                     observed_at="2025-04-30T14:08:10+09:00",
                     latitude=34.965, longitude=136.624, grid=True)
    image = cv2.imread(str(Path(folder)/"20250430140810_annotated.png"))
    cx, cy = result["disk_center"]
    margin = result["disk_radius"] + 50
    x1, y1 = max(0, cx-margin), max(0, cy-margin)
    x2, y2 = min(image.shape[1], cx+margin), min(image.shape[0], cy+margin)
    examples = root / "examples"
    examples.mkdir(exist_ok=True)
    if not cv2.imwrite(str(examples/"20250430140810_grid.jpg"), image[y1:y2, x1:x2],
                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError("Cannot save grid example")
    result["display_crop_xyxy"] = [x1, y1, x2, y2]
    result["site_precision"] = "approximate city-centre coordinates, not exact capture site"
    (examples/"20250430140810_grid_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
