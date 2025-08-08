import os
from dotenv import load_dotenv
from agents import AsyncOpenAI, OpenAIChatCompletionsModel, Agent, Runner, set_tracing_disabled
import chainlit as cl


# Load environment variables
load_dotenv()
set_tracing_disabled(True)



# Chat start event
@cl.on_chat_start
async def start():
    MODEL_NAME = "gemini-2.0-flash"
    API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not API_KEY:
        raise ValueError("🔐 GEMINI_API_KEY environment variable is not set.")
    
    # External client setup
    external_client = AsyncOpenAI(
        api_key=API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    
    # Model setup
    model = OpenAIChatCompletionsModel(
        model=MODEL_NAME,
        openai_client=external_client,
    )
    


    
    # Agent setup
    assistant = Agent(
        name="Study Planner Assistant 🇵🇰",
        instructions="""You are a helpful assistant about  Study Schedule Planner. 
        if the user ask about anything not related about  Study Schedule Planner
        "I'm sorry😔, I can only help with topics related to  Study Schedule Planner."
        """,
        model=model,
    )
    
    cl.user_session.set("agent", assistant)
    cl.user_session.set("chat_history", [])

    await cl.Message(content="👋 Welcome to the Assistant! Ask me anything about  Study Schedule Planner 🇵🇰.").send()

# Message event handler
@cl.on_message
async def main(message: cl.Message):
    msg = await cl.Message(content = " 💭 Please wait....").send()

    assistant = cl.user_session.get("agent")
    history = cl.user_session.get("chat_history") or []

    history.append({"role": "user", "content": message.content})

    result = await Runner.run(
        starting_agent=assistant,
        input=history,
    )

    msg.content = result.final_output + " 😊"
    await msg.update()

    cl.user_session.set("chat_history", result.to_input_list())
    result.final_output
