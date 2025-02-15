import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv('OPENAI_KEY')

# call openai skeleton
import openai

def call_openai(prompt):
    # openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"]

# if __name__ == "__main__":
#     api_key = "your_openai_api_key_here"
#     prompt = "Tell me a fun fact about space."
#     output = call_gpt4o_mini(prompt, api_key)
#     print(output)
