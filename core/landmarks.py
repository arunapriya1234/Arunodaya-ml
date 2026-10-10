import numpy as np

def normalize_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
    """
    Normalizes 21 3D hand keypoints for translation and scale invariance.
    
    Input: Array of shape (21, 3) representing (x, y, z) coordinates.
    Output: Flattened 1D array of shape (63,) normalized.
    """
    if landmarks_3d is None or len(landmarks_3d) != 21:
        return np.zeros(63, dtype=np.float32)

    coords = np.array(landmarks_3d, dtype=np.float32)

    # 1. Translation Invariance: Shift wrist (landmark 0) to origin (0, 0, 0)
    wrist = coords[0].copy()
    coords -= wrist

    # 2. Scale Invariance: Scale by distance from wrist (0) to middle finger MCP (9)
    dist = np.linalg.norm(coords[9])
    if dist > 1e-6:
        coords /= dist

    # Flatten (21, 3) into (63,)
    return coords.flatten()

def normalize_multi_hand_landmarks(all_coords, max_hands: int = 2) -> np.ndarray:
    """
    Normalizes up to max_hands (default 2) hands into a single concatenated vector.
    Each hand occupies 63 dimensions (total 126 for 2 hands).
    If fewer hands are detected, remaining slots are zero-padded.
    Hands are sorted by wrist x-coordinate for consistent left-to-right ordering.
    """
    vector_len = max_hands * 63
    if all_coords is None or len(all_coords) == 0:
        return np.zeros(vector_len, dtype=np.float32)

    # Sort detected hands horizontally by wrist X coordinate (landmark 0)
    sorted_hands = sorted(all_coords, key=lambda c: c[0][0])[:max_hands]

    feats = []
    for hand_coords in sorted_hands:
        norm_hand = normalize_landmarks(hand_coords)
        feats.append(norm_hand)

    # Pad missing hands with zeros
    while len(feats) < max_hands:
        feats.append(np.zeros(63, dtype=np.float32))

    return np.concatenate(feats).astype(np.float32)

def augment_landmarks(features: np.ndarray, jitter_std: float = 0.02) -> np.ndarray:
    """
    Applies subtle Gaussian jitter to non-zero landmark coordinates for data augmentation.
    """
    noisy = features.copy()
    mask = (noisy != 0.0)
    noise = np.random.normal(0, jitter_std, size=features.shape).astype(np.float32)
    noisy[mask] += noise[mask]
    return noisy
