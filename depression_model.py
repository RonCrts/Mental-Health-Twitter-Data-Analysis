import requests

class DepressionModel:
    def __init__(self):
        self.API_URL = "https://api-inference.huggingface.co/models/rafalposwiata/deproberta-large-depression"
        self.headers = {"Authorization": "Your API key"}

    def predict(self, text):
        payload = {"inputs": text}
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.json()

