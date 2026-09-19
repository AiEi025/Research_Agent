import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def model(tune:int=0):
    llm_openai = ChatOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    temperature=tune,
    timeout= 30)
    return llm_openai