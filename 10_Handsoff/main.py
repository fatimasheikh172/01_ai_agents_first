import os
from dotenv import load_dotenv
from agents import Agent , OpenAIChatCompletionsModel , RunConfig , Runner , AsyncOpenAI 
import asyncio


load_dotenv() # Load environment variables from .env file

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

translater_agent = Agent(
    name = "Translater Agent",
    instructions="Translate whatever the user gives.",
    handoff_description="Translate the text"
)

summary_agent = Agent(
    name = "Summary Agent",
    instructions="Summarize whatever the user gives.",
    handoff_description="Summarize the text"
)


triage_agent = Agent(
    name = "Triage Agent",
    instructions="If the user says any text and wants it translated, then translate it. If the user asks for a summary of something, then provide that",
    handoffs=[translater_agent, summary_agent]
)


async def main():
    result = await Runner.run(triage_agent, 'translate "Whats your name" into Arabic and give me a summary about Pakistan', run_config=config)
    print("Final Response: ",result.final_output)

if __name__ == "__main__":
    asyncio.run(main())