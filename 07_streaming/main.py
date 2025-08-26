import asyncio
import os
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent 
from agents import Agent, Runner, OpenAIChatCompletionsModel

# ------------------- Load Environment Variables -------------------

load_dotenv(find_dotenv())

# ------------------- API Configuration -------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  
MODEL = "gemini-2.0-flash"                    
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/" 


client = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url=BASE_URL)



model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)


agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model=model,
)

# ------------------- Main Async Function -------------------
async def main():
    # Start a streamed conversation with the agent
    result = Runner.run_streamed(
        starting_agent=agent, 
        input="tell me something interesting about pakistan in 1000 words"
    )
    
    # Stream and print the AI's text as it comes in
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end='', flush=True)


if __name__ == "__main__":
    asyncio.run(main())
