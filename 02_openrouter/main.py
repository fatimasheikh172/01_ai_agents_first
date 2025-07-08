
import asyncio
import os
from openai import AsyncOpenAI  # type: ignore
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled # type: ignore
from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") 


BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-r1-0528:free"

print(f"Loaded API Key: {OPENROUTER_API_KEY is not None}")

client = AsyncOpenAI(
    api_key = OPENROUTER_API_KEY,
    base_url = BASE_URL
)

set_tracing_disabled(disabled=True)

async def main():
    # This agent will use the custom LLM provider
    agent = Agent(
        name = "Fatima's Assistant",
        instructions= "You only respond like a firendly teacher ",
        model=OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
    )
    
    result = await Runner.run(
        agent,
        "what is English , tell me uses and importance in now a days?"
    )
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())    