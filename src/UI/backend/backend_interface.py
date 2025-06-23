import requests
import logging

class BackendInterface:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def ping(self):
        response = requests.get(f"{self.backend_url}/ping")
        return response.json()

    def get_ocr(self, file_name: str, file_bytes: bytes):
        files = {'image': (file_name, file_bytes)}
        response = requests.post(f"{self.backend_url}/OCR", files=files)
        return response.json()


    def calculate(self,query: str):
        response = requests.post(f"{self.backend_url}/calculator", params={"query": query})
        return response.json()

    def json_format(self,query: str = None):
        response = requests.post(f"{self.backend_url}/json_formatter", params={"query": query})
        return response.json()

    def translator(self, text: str):
        response = requests.post(f"{self.backend_url}/translator", params={"text": text})
        return response.json()

    def infer(self, input_text: str):
         # MOCKED: Randomly return a classify label
        # random_class = random.choice([ "translate", "calculator", "json_formatter"])
        # return {
        #     "classify": random_class,
        #     "processed_text": input_text
        # }
        try:
            payload = {"input_text": input_text}
            response = requests.post(f"{self.backend_url}/infer_t5", json=payload)
            return response.json()
        except Exception as e:
            return {"error": str(e)}    

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
        self.logger.info(f"User Input: {user_input}")
        if user_input.lower() == "ping":
            response=self.ping()
            return response.get("response","")
        # call the infer endpoint to classify the user input
        classify_response = self.infer(user_input)
        self.logger.info(f"Classify Response: {classify_response}")
        response=classify_response.get("response", "")
        self.logger.info(f"Response from classify: {response}")
        #call tool endpoints based on classification tool in string
        if "calculate" in response.lower():
            response = self.calculate(response)
        elif "json" in response.lower():
            response = self.json_format(response)
        elif "translator" in response.lower():
            response = self.translator(response)
        elif "ocr" in response.lower() and file_bytes is not None and file_name is not None:
            response = self.get_ocr(file_name, file_bytes)
        else:
            response = {"response": "No valid tool found for the input."}
        return response.get("response", "")
