import os
from dotenv import load_dotenv
from typing import cast 
import chainlit as cl
from agents import Agent , Runner , OpenAIChatCompletionsModel , AsyncOpenAI 
from agents.run import RunConfig

# Load environment variables from .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.") # Raise an error if the value is invalid

@cl.on_chat_start  # Decorator to handle chat start event
async def start():

    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    model = OpenAIChatCompletionsModel(
        openai_client=external_client,  # Use the external client for OpenAI API
        model="gemini-2.0-flash",
    )

    config = RunConfig(  # RunConfig decide karta hai ke is run me agent ke rules aur options kya honge
        model=model,
        model_provider=external_client,
        tracing_disabled=True,  # Disable tracing for this run
    )

    cl.user_session.set("chat_history" , [])  # Initialize chat history in user session
    cl.user_session.set("config", config)  # Store the run configuration in user session

    agent = Agent(
        name="English Tutor",
        instructions="you are a helpful English tutor. You will help the user with their English questions and provide explanations.",
        model=model
    )

    cl.user_session.set("agent", agent)  # Store the agent in user session
    await cl.Message(
        content="Welcome to the English Tutor!",
    ).send()  # Send a welcome message to the user

@cl.on_message  # Decorator to handle incoming messages
async def main(message: cl.Message):
    msg = cl.Message(content="Thinking...")
    await msg.send()  # Send a "Thinking..." message to the user

    agent: Agent  = cast(Agent , cl.user_session.get("agent"))  # Retrieve the agent from user session
    config:RunConfig = cast(RunConfig, cl.user_session.get("config"))  # Retrieve the run configuration from user session
    history = cl.user_session.get("chat_history") or []  # Retrieve chat history from user session
    history.append({"role": "user", "content": message.content})  # Append the user's message to chat history

    try:
        print("\n \nRunning agent with message:",history)  # Print the chat history for debugging

        result = Runner.run_sync(
            starting_agent=agent,  # Start the agent with the provided configuration
            input=history,  # Input is the chat history
            run_config=config,  # Run configuration
        )

        responce_content = result.final_output  # Get the final output from the result
        msg.content = responce_content  # Update the message content with the response
        await msg.update()  # Update the message in the chat

        cl.user_session.set("chat_history", result.to_input_list())  
         

    except Exception as e:
            msg.content = f"Error: {str(e)}"  # If an error occurs, update the message content with the error message
            await msg.update()  # Update the message in the chat
            print(f"Error: {str(e)}")  # Print the error message for debugging





