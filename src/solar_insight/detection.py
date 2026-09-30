"""Detection routines extracted from the supplied new002 prototype.

Pixel thresholds and seed-based grouping preserve legacy behavior.
These are candidate detections, not verified Zurich classifications.
"""
import cv2
import numpy as np

def preprocess_for_detection(gray):
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    blurred = cv2.GaussianBlur(sharpened, (3, 3), 0)
    return blurred

def detect_solar_disk(gray):
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("太陽円の輪郭が見つかりませんでした")
    largest = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest)
    return (int(x), int(y)), int(radius)

def detect_sunspots_from_masked(masked_img, center, radius):
    binary = cv2.adaptiveThreshold(masked_img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    interior = np.zeros_like(binary)
    cv2.circle(interior, center, max(0, radius - 15), 255, -1)
    binary = cv2.bitwise_and(binary, interior)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers, boxes = [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 3000:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                dist = np.hypot(cx - center[0], cy - center[1])
                if dist < radius - 15:
                    centers.append((cx, cy))
                    x, y, w, h = cv2.boundingRect(cnt)
                    boxes.append((x, y, w, h))
    return centers, boxes

def group_sunspots_with_labels(centers):
    groups = []
    used = set()
    for i, c1 in enumerate(centers):
        if i in used:
            continue
        group = [c1]
        used.add(i)
        for j, c2 in enumerate(centers):
            if j not in used and np.linalg.norm(np.array(c1) - np.array(c2)) < 80:
                group.append(c2)
                used.add(j)
        groups.append(group)
    return groups
