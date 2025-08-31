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

math_agent = Agent(
    name = "Math Agent",
    instructions="Solve whatever math problem the user gives.",
    handoff_description="Solve the math problem"
)

english_agent = Agent(
    name = "English Agent",
    instructions="Correct whatever English text the user gives.",
    handoff_description="Correct the English text"  
)

urdu_agent = Agent(
    name = "Urdu Agent",
    instructions="solve whatever Urdu text the user gives. in urdu",
    handoff_description="Give the answer in Urdu"
)

triage_agent = Agent(
    name = "Triage Agent",
    instructions="If the user says any math problem, then solve it. If the user says any English text, then correct it. If the user says any Urdu text, then solve it in Urdu.",
    handoffs=[math_agent, english_agent, urdu_agent]
)

async def main():
    result = await Runner.run(triage_agent,
            'What is 12*13? Also correct this sentence "He go to school every day". Also solve this in Urdu "پاکستان کا دارالحکومت کیا ہے؟"',
             run_config=config)
    print("Final Response: ",result.final_output)


if __name__ == "__main__":
    asyncio.run(main()) 
