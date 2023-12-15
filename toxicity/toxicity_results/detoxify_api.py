from detoxify import Detoxify
from typing import Dict, Optional, List

class DetoxifyScorer:
    def __init__(self):
        self.model = Detoxify('original', device='cuda:2')
    def get_scores(self, input_text: str) -> Dict[str, float]:
        return self.model.predict(input_text)

