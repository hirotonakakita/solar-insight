"""Experimental legacy inference; outputs are uncalibrated model scores."""
import json
import math
from PIL import Image
import torch
from torchvision import transforms
from .model import FlarePredictionModel


def predict(image_path, features, checkpoint, scaler_path):
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    names = ["group_count", "spot_count", "relative_number"]
    if scaler.get("feature_order") != names:
        raise ValueError("Scaler feature order must be group_count, spot_count, relative_number")
    mean, scale = scaler["mean"], scaler["scale"]
    if len(mean) != 3 or len(scale) != 3 or any(not math.isfinite(v) for v in mean + scale) or any(v <= 0 for v in scale):
        raise ValueError("Invalid scaler parameters")
    numeric = torch.tensor([[(features[n]-m)/s for n,m,s in zip(names,mean,scale)]], dtype=torch.float32)
    with Image.open(image_path) as img:
        image = transforms.ToTensor()(img.convert("RGB")).unsqueeze(0)
    model = FlarePredictionModel()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        outputs = model(image, numeric)
    return {"x_score": torch.sigmoid(outputs["x_occurred"]).item(),
            "m_score": outputs["m_occurred"].item(),
            "m_count_raw": outputs["m_count"].item(),
            "validated_forecast": False,
            "note": "Legacy same-day labels; preprocessing and transfer to observed images are unvalidated"}
