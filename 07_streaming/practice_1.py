import os
from dotenv import load_dotenv  
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig    
from openai.types.responses import ResponseTextDeltaEvent
import asyncio


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")  
if not gemini_api_key:                  
    raise ValueError("GEMINI_API_KEY environment variable not set")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,     
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,  
    model="gemini-2.5-flash"
)

agent = Agent(
    name="Fatima's Assistant",
    instructions="You are a helpful for hijabs and abayas guide assistant for choose the best hijab and abaya , fabric, color, style, and occasion.",
    model=model,
)

config = RunConfig(
    model=model,    
    model_provider=external_client,
    tracing_disabled=True
)

async def main():
    result = Runner.run_streamed(
        starting_agent=agent,
        input="Suggest me the best hijab and abaya for wedding party"
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data ,ResponseTextDeltaEvent):
            print(event.data.delta , end= '', flush=True)

if __name__ == "__main__":
    asyncio.run(main())


    
