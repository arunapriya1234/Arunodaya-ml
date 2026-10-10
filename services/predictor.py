import os
import json
import torch
import numpy as np
from core.model import LandmarkMLP
from core.landmarks import normalize_landmarks, normalize_multi_hand_landmarks

class GesturePredictor:
    """
    Inference Engine: Loads model weights and metadata, then evaluates live hand landmarks
    for real-time Mudra / Gesture Classification.
    """
    def __init__(self, model_dir: str = "outputs/models/mudra_mlp", confidence_threshold: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        meta_path = os.path.join(model_dir, "meta.json")
        model_path = os.path.join(model_dir, "model.pt")
        
        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.meta = {"input_dim": 126, "classes": ["UNKNOWN"]}
            
        self.classes = self.meta.get("classes", ["UNKNOWN"])
        self.input_dim = self.meta.get("input_dim", 126)

        # Load PyTorch model
        self.model = LandmarkMLP(input_dim=self.input_dim, num_classes=len(self.classes))
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded predictor model weights from: {model_path} ({len(self.classes)} classes)")
        else:
            print(f"Warning: Model file {model_path} not found. Operating with un-trained weights.")

        self.model.to(self.device)
        self.model.eval()

    def predict(self, hand_coords):
        """
        Predicts mudra class from hand coordinates.
        Accepts:
          - all_coords: list of numpy arrays, one per detected hand [(21, 3), ...]
          - or single hand array: shape (21, 3)
        Returns: (predicted_class_name, confidence, top3_list)
        """
        if hand_coords is None:
            return "NO HAND DETECTED", 0.0, []

        # Convert single hand format to list if needed
        if isinstance(hand_coords, np.ndarray) and hand_coords.shape == (21, 3):
            hand_list = [hand_coords]
        elif isinstance(hand_coords, list) and len(hand_coords) > 0:
            hand_list = hand_coords
        else:
            return "NO HAND DETECTED", 0.0, []

        # Feature normalization based on model input dimension
        if self.input_dim == 126:
            feat = normalize_multi_hand_landmarks(hand_list, max_hands=2)
        else:
            feat = normalize_landmarks(hand_list[0])

        tensor_x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor_x)
            probabilities = torch.softmax(logits, dim=1)[0]
            conf, pred_idx = torch.max(probabilities, dim=0)

            # Top 3 candidates
            topk_vals, topk_indices = torch.topk(probabilities, k=min(3, len(self.classes)))
            top3 = [(self.classes[idx.item()], float(val.item())) for val, idx in zip(topk_vals, topk_indices)]

        confidence = conf.item()
        pred_label = self.classes[pred_idx.item()]

        if confidence < self.confidence_threshold:
            return "UNCERTAIN", confidence, top3

        return pred_label, confidence, top3
