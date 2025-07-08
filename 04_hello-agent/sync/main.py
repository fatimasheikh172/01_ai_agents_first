from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Get the Gemini API key from environment
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# Step 2: Create the external OpenAI-compatible client
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Step 3: Define the model using the client
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

# Step 4: Configure how the model will run
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Step 5: Define your agent with instructions
agent = Agent(
    name="Assistant",
    instructions="Create a informative story for child in easy english  child understand to words"
)

# Step 6: Run the agent synchronously with a user message
response = Runner.run_sync(agent, "Hello!", run_config=config)

# Step 7: Print the result
print(response.final_output)
