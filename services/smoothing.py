from collections import deque, Counter
import time

class PredictionSmoother:
    """
    Smooths predictions across consecutive video frames using majority voting
    and cooldown controls to eliminate rapid UI jitter.
    """
    def __init__(self, buffer_size: int = 7, cooldown_seconds: float = 0.3):
        self.buffer = deque(maxlen=buffer_size)
        self.cooldown_seconds = cooldown_seconds
        self.last_state_change_time = 0.0
        self.current_stable_prediction = "NO HAND DETECTED"

    def update(self, raw_prediction: str) -> str:
        """
        Adds new frame prediction to buffer and calculates stable majority decision.
        """
        self.buffer.append(raw_prediction)
        
        # Calculate majority prediction in buffer window
        counts = Counter(self.buffer)
        majority_prediction, majority_count = counts.most_common(1)[0]
        
        # Require at least 50% consensus in buffer
        threshold = len(self.buffer) // 2 + 1
        new_candidate = majority_prediction if majority_count >= threshold else "UNCERTAIN"

        # Apply cooldown check before switching prediction state
        now = time.time()
        if new_candidate != self.current_stable_prediction:
            if (now - self.last_state_change_time) >= self.cooldown_seconds:
                self.current_stable_prediction = new_candidate
                self.last_state_change_time = now

        return self.current_stable_prediction
