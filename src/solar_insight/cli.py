"""CLI: image analysis works without machine learning dependencies."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from .detection import (preprocess_for_detection, detect_solar_disk,
                        detect_sunspots_from_masked, group_sunspots_with_labels)


def analyze(image_path, output_dir):
    image_path, output_dir = Path(image_path), Path(output_dir)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if min(gray.shape) < 32:
        raise ValueError("Image dimensions must be at least 32 pixels")
    center, radius = detect_solar_disk(gray)
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)
    detection_image = cv2.bitwise_and(preprocess_for_detection(gray), mask)
    centers, boxes = detect_sunspots_from_masked(detection_image, center, radius)
    groups = group_sunspots_with_labels(centers)
    result = {"image": image_path.name, "disk_center": list(center),
              "disk_radius": radius, "spot_count": len(centers),
              "group_count": len(groups),
              "relative_number": 10 * len(groups) + len(centers)}
    annotated = image.copy()
    for cx, cy in centers:
        cv2.drawMarker(annotated, (cx, cy), (0, 255, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=10)
    for idx, group in enumerate(groups, 1):
        points = np.array(group)
        x1, y1 = np.maximum(points.min(axis=0) - 10, 0)
        x2, y2 = np.minimum(points.max(axis=0) + 10, [gray.shape[1]-1, gray.shape[0]-1])
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 1)
        cv2.putText(annotated, str(idx), (int(x1), max(12, int(y1)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
    canvas = np.hstack([image, annotated])
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / image_path.stem
    if not cv2.imwrite(str(base) + "_comparison.png", canvas):
        raise OSError("Failed to write comparison image")
    Path(str(base) + "_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with Path(str(base) + "_spots.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "box_x", "box_y", "box_width", "box_height"])
        for point, box in zip(centers, boxes):
            writer.writerow([*point, *box])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scaler", type=Path)
    parser.add_argument("--allow-legacy-preprocessing", action="store_true")
    args = parser.parse_args()
    if args.checkpoint and (not args.scaler or not args.allow_legacy_preprocessing):
        parser.error("Experimental inference needs --scaler and --allow-legacy-preprocessing")
    try:
        result = analyze(args.image, args.output_dir)
        if args.checkpoint:
            from .inference import predict
            result["experimental_inference"] = predict(args.image, result, args.checkpoint, args.scaler)
            (args.output_dir / f"{args.image.stem}_summary.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8")
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
