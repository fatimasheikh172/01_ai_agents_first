from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    api_key = os.getenv("GEMINI_API_KEY")

    responce = completion(
       model="gemini/gemini-1.5-flash",
        messages=[{"content": "explain pakistan in easy  english?","role": "user"}]
    )

    print(responce['choices'][0]['message']['content'])

if __name__ == "__main__":
    main()