import requests

class BiasDetector:
    def __init__(self):
        self.API_URL = "https://api-inference.huggingface.co/models/amedvedev/bert-tiny-cognitive-bias"
        self.headers = {"Authorization": "Bearer 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"}

    def predict(self, text):
        payload = {"inputs": text}
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()

