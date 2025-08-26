import os 
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel ,  AsyncOpenAI
from agents.run import RunConfig
import chainlit as cl
from typing import cast

# ------------------- Load Environment Variables -------------------

load_dotenv()

# ------------------- API Configuration -------------------

gemini_api_key = os.getenv("GEMINI_API_KEY")  
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set") 

@cl.on_chat_start

async def start():

    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )   

    model = OpenAIChatCompletionsModel(
        openai_client=external_client,
        model="gemini-2.5-flash"
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True

    )

    agent = Agent(
        name="Travel Assistant",
        instructions="""You are a helpful travel assistant who helps users plan trips and provides travel recommendations. 
        You are friendly and engaging. You always ask questions to better understand the user's preferences and needs.""",
        model=model,
    )

    cl.user_session.set("agent", agent)

    await cl.Message(content="Hello! I'm your travel assistant. How can I help you plan your next trip?").send()

   
@cl.on_message

async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Retrieve the chat history from the session.
    history = cl.user_session.get("chat_history") or []

    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    # Create a new message object for streaming
    msg = cl.Message(content="")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    try:
        print("\n[CALLING_AGENT_WITH_CONTEXT]\n", history, "\n")
        # Run the agent with streaming enabled
        result = Runner.run_streamed(agent, history, run_config=config)

        # Stream the response token by token
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

        # Append the assistant's response to the history.
        history.append({"role": "assistant", "content": msg.content})

        # Update the session with the new history.
        cl.user_session.set("chat_history", history)

      

    except Exception as e:
        await msg.update(content=f"Error: {str(e)}")
        print(f"Error: {str(e)}")
