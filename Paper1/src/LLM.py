import instructor
import vertexai
from openai import OpenAI
from google.auth import default, transport

class GoogleLLM:
    def __init__(self):
        self.name = "Gemini_1.5_pro"
        self.PROJECT_ID = "gen-lang-client-0539303742"
        self.MODEL_ID = "google/gemini-1.5-pro-001"
        self.location = "us-central1"
        vertexai.init(project=self.PROJECT_ID, location=self.location)
    
    def reAuth(self):
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_request = transport.requests.Request()
        credentials.refresh(auth_request)
        
        gemini_base_url=f"https://{self.location}-aiplatform.googleapis.com/v1beta1/projects/{self.PROJECT_ID}/locations/{self.location}/endpoints/openapi"
        
        client = instructor.from_openai(
            OpenAI (
                base_url=gemini_base_url,
                api_key=credentials.token
            ),
            mode=instructor.Mode.JSON,
        )

        self.client = client
    
    def llmResponse(self, temperature, instr, prompt, responseModel):
        response = self.client.chat.completions.create(
            model=self.MODEL_ID,
            max_completion_tokens=8192,
            temperature=temperature,
            messages=[
                { "role": "system", "content": instr},
                { "role": "user", "content": prompt}
            ],
            response_model=responseModel,
        )

        return response