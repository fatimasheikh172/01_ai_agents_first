from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, set_tracing_disabled, function_tool
import os
from dotenv import load_dotenv
import requests
import random

load_dotenv()  # Load environment variables from .env file
set_tracing_disabled(disabled=True)  # Disable tracing globally

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


@function_tool
def how_many_jokes():
    """
    Gets random number of jokes
    """
    return random.randint(1, 10)


@function_tool
def get_weather(city: str):
    """
    Gets the current weather for a given city
    """
    try:
        result = requests.get(
            f"https://api.weatherapi.com/v1/current.json?key=8e3aca2b91dc4342a1162608252604&q={city}"
        )
        data = result.json()
        return f"The current weather in {city} is {data['current']['temp_c']}°C with {data['current']['condition']['text']}."
    except Exception as e:
        return f"Failed to get weather data: {str(e)}"


agent = Agent(
    name="Agent Assistant",
    instructions=""" 
    Use the tool to determine how many jokes to include in your response.
    also provide the current weather in using the get_weather tool.""",
    model=model,
    tools=[how_many_jokes, get_weather] 
)

result = Runner.run_sync(
    agent,
    input="what is the wather in karachi and tell me some jokes"
)

print(result.final_output)
