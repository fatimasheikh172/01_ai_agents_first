import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel ,input_guardrail, GuardrailFunctionOutput , InputGuardrailTripwireTriggered
from agents.run import RunConfig
import asyncio
from pydantic import BaseModel



load_dotenv()

class MathHomeWorkOutput(BaseModel):
    is_math_work:bool
    reasoning:str

class EnglishHomeWorkOutput(BaseModel):
    is_math_work:bool
    reasoning:str


gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")



external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)


inputGuardrialAgent = Agent(
    name= "input guardrial agent",
    instructions="you have to check user quires is related to math or not",
    output_type=MathHomeWorkOutput,
    model=model

)

@input_guardrail
async def math_guardrial(ctx , agent , input):
    print("prompt: " + " " + input)
    result = await Runner.run(inputGuardrialAgent , input)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_Math_work
    )


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You are helpful Assistent.",
        model=model
    )

    result = await Runner.run(agent, "Tell me about python in programming.", run_config=config)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())