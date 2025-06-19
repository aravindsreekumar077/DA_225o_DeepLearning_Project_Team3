import requests

class BackendInterface:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url

    def ping(self):
        response = requests.get(f"{self.backend_url}/ping")
        return response.json()

    def get_ocr(self, image_path: str):
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            response = requests.post(f"{self.backend_url}/OCR", files=files)
        return response.json()

    def calculate(self):  
        response = requests.post(f"{self.backend_url}/calculator")
        return response.json()

    def json_format(self): 
        response = requests.post(f"{self.backend_url}/json_formatter")
        return response.json()

    def translator(self, text: str):
        response = requests.post(f"{self.backend_url}/translator", params={"text": text})
        return response.json()
