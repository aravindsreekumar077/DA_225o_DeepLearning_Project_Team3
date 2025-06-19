import requests
import random
class BackendInterface:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url

    def ping(self):
        response = requests.get(f"{self.backend_url}/ping")
        return response.json()

    def get_ocr(self, file_name: str, file_bytes: bytes):
        files = {'image': (file_name, file_bytes)}
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

    def infer(self, input_text: str):
        # payload = {"input_text": input_text}
        # response = requests.post(f"{self.backend_url}/infer", json=payload)
        # return response.json()
         # MOCKED: Randomly return a classify label
        random_class = random.choice([ "translate", "calculator", "json_formatter"])
        return {
            "classify": random_class,
            "processed_text": input_text
        }

    def get_agent_response(self, user_input: str, file_bytes: bytes = None, file_name: str = None):
        """
        Main entry point: classify the user input via /infer,
        then route to the appropriate backend tool.

        Parameters:
            user_input (str): the raw input from user
            file_path (str): optional file path for OCR if required

        Returns:
            dict: response from the selected backend tool
        """
        classification = self.infer(user_input)

        classify = classification.get("classify", "").lower()
        processed_text = classification.get("processed_text", user_input)

        if classify == "ocr":
            if file_path is None:
                return {"error": "OCR requires an image file."}
            return self.get_ocr(file_path)

        elif classify == "translate":
            return self.translator(processed_text)

        elif classify == "calculator":
            return self.calculate()

        elif classify == "json_formatter":
            return self.json_format()

        else:
            return {"error": f"Unknown classification from /infer: '{classify}'"}
