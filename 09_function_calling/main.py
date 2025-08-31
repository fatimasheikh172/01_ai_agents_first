import os
from dotenv import load_dotenv
from agents import Agent , Runner , OpenAIChatCompletionsModel , function_tool , RunConfig , AsyncOpenAI

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")


external_client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model = "gemini-2.5-flash"
)

config = RunConfig(
  model=model,
  model_provider=external_client,
  tracing_disabled=True
)

@function_tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b    


agent = Agent(
    name="MathAgent",
    model=model,
    tools=[add]   
)

# Step 3: Run the agent
result = Runner.run_sync(
    agent,
    input="What is 12 plus 8?"
)

print(result.final_output)