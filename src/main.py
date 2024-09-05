import os
from typing import Optional
from mlp_sdk.hosting.host import host_mlp_cloud
from pydantic import BaseModel, validator
from mlp_sdk.abstract import Task

import os
import requests
from pydantic import BaseModel, validator

MLP_API_KEY = os.getenv('MLP_API_KEY')
BASE_API_URL = 'https://caila.io/api/mlpgate/account'


class RequestData(BaseModel):
    account: Optional[str] = 'just-ai'
    service: str
    model: str
    text: str

    @validator('text')
    def text_must_be_under_1000_chars(cls, v):
        if len(v) > 1000:
            raise ValueError('Text must be under 1000 characters')
        return v


class PredictResponse(BaseModel):
    response: str

    def __init__(self, response):
        super().__init__(response=response)


class SimpleActionExample(Task):
    
    def __init__(self, config: BaseModel, *args):
        self.api_key = MLP_API_KEY
        
    def _get_api_url(self, account: str, service: str) -> str:
        return f'{BASE_API_URL}/{account}/model/{service}/predict'

    def _get_headers(self) -> dict:
        return {
            "MLP-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

    def _get_request_payload(self, model: str, text: str) -> dict:
        return {
            "messages":[{"role":"user","content": f'Найди все ФИО в тексте. Выведи их в формате list вида ["ln1", "ln2", ...]: {text}'}],
            "model": model
        }   
    
    def predict(self, data: RequestData, config: BaseModel) -> PredictResponse:
        api_url = self._get_api_url(data.account, data.service)
        headers = self._get_headers()
        request_payload = self._get_request_payload(data.model, data.text)
        
        response = requests.post(api_url, headers=headers, json=request_payload)
        
        if response.status_code == 200:
            response_data = response.json()
            
            message_content = response_data["choices"][0]["message"]["content"]
            
            return PredictResponse(response=message_content)
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")


if __name__ == "__main__":
    host_mlp_cloud(SimpleActionExample, BaseModel())