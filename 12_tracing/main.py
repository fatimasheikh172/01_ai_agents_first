import os
from dotenv import load_dotenv
from agents import Agent , Runner , OpenAIChatCompletionsModel, AsyncOpenAI, set_trace_processors
from agents.tracing.processors import ConsoleSpanExporter , BatchTraceProcessor 
load_dotenv()

exporter = ConsoleSpanExporter()
processor = BatchTraceProcessor(exporter)

set_trace_processors([processor])

gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key= gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client,
    model = "gemini-2.5-flash" 
)

agent = Agent(
    name = "Assistant",
    instructions = "You are a helpful assistant that translates English to French.",
    model = model
)

result = Runner.run_sync(
    starting_agent=agent,
    input="Translate the following English text to French: 'Hello, how are you?"
)

print(result.final_output)