import os
from dotenv import load_dotenv
load_dotenv()

# model = "gpt-4-0613"
model = "gpt-4-1106-preview"
api_key = os.getenv('OPENAI_API_KEY')

llm_config = {
    "timeout": 600,
    "seed": 42,
    "model": model,  # make sure the endpoint you use supports the model
    "temperature": 0,
    "api_key": api_key
}