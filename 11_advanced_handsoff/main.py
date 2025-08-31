import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, handoff
from agents.run import RunConfig

# Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Define two simple agents
support_agent = Agent(
    name="Support Agent",
    model=OpenAIChatCompletionsModel(model="gpt-4", api_key=api_key)
)

sales_agent = Agent(
    name="Sales Agent",
    model=OpenAIChatCompletionsModel(model="gpt-4", api_key=api_key)
)

# Define handoff rule
handoff_rule = handoff(
    from_agent=support_agent,           # Source Agent
    to_agent=sales_agent,               # Destination Agent
    input_type="text",                  # Only handle text inputs
    input_filter=lambda msg: "buy" in msg.lower(),  # Only trigger if msg contains "buy"
    is_enabled=lambda: True,            # Always enabled
    on_handoff=lambda msg: print(f"[HANDOFF TRIGGERED] User said: '{msg}' → Moving Support → Sales")
)

# Attach handoff rule to support_agent
support_agent.add_handoff(handoff_rule)

# Runner config
config = RunConfig()

# Example Run
result = Runner.run_sync(
    agent=support_agent,
    input="I want to buy a new laptop",   # 👈 This input will trigger handoff
    run_config=config
)

# Print the final result
print("Final Output:", result.output_text)
