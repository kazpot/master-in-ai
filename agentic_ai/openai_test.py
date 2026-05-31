import os
from openai import OpenAI
from enum import Enum
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=httpx.Client()
)

class OpenAIModel(str, Enum):
        # GPT‑4o ("Omni") – flagship multimodal model released May 13, 2024.
        # Handles text, audio, images, and video with native voice/vision support;
        # ~128 K token context window.
        GPT_4O = "gpt-4o"

        # GPT‑4o‑mini – smaller, cost-effective variant released July 18, 2024.
        # Multimodal like GPT‑4o, same context window, optimized for affordability.
        GPT_4O_MINI = "gpt-4o-mini"

        # GPT‑4.1 – developer-focused flagship released April 14, 2025.
        # Massive 1 M token context window; excels at coding, reasoning, instruction-following.
        GPT_41 = "gpt-4.1"

        # GPT‑4.1‑mini – compact variant in the 4.1 family.
        # Same 1 M context, much lower latency and cost; matches or outperforms GPT‑4o on many benchmarks.
        GPT_41_MINI = "gpt-4.1-mini"

        # GPT‑4.1‑nano – smallest, fastest 4.1 variant.
        # 1 M token context, highly efficient and cost-effective for lightweight tasks.
        GPT_41_NANO = "gpt-4.1-nano"


MODEL = OpenAIModel.GPT_41_NANO

def get_completion(system_prompt, user_prompt, model=MODEL):
    """
    Function to get a completion from the OpenAI API.
    Args:
        system_prompt: The system prompt
        user_prompt: The user prompt
        model: The model to use (default is gpt-4.1-mini)
    Returns:
        The completion text
    """
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"


def display_responses(*args):
    """Helper function to display responses, works in both Jupyter and terminal."""
    try:
        from IPython import get_ipython
        from IPython.display import display, Markdown
        if get_ipython() is not None:
            markdown_string = "<table><tr>"
            for arg in args:
                markdown_string += f"<th>System Prompt:<br />{arg['system_prompt']}<br /><br />"
                markdown_string += f"User Prompt:<br />{arg['user_prompt']}</th>"
            markdown_string += "</tr><tr>"
            for arg in args:
                markdown_string += f"<td>Response:<br />{arg['response']}</td>"
            markdown_string += "</tr></table>"
            display(Markdown(markdown_string))
            return
    except ImportError:
        pass
    for arg in args:
        print("-" * 60)
        print(f"System Prompt: {arg['system_prompt']}")
        print(f"User Prompt:   {arg['user_prompt']}")
        print(f"Response:\n{arg['response']}")
    print("-" * 60)

# No changes needed in this cell
plain_system_prompt = "You are a helpful assistant."  # A generic system prompt
user_prompt = "Give me a simple plan to declutter and organize my workspace."

print(f"Sending prompt to {MODEL} model...")
baseline_response = get_completion(plain_system_prompt, user_prompt)
print("Response received!\n")

display_responses(
    {
        "system_prompt": plain_system_prompt,
        "user_prompt": user_prompt,
        "response": baseline_response,
    }
)

role_system_prompt = "You are a certified professional organizer"

print("Sending prompt with professional role...")
role_response = get_completion(role_system_prompt, user_prompt)
print("Response received!\n")

# Show last two prompts and responses
display_responses(
    {
        "system_prompt": plain_system_prompt,
        "user_prompt": user_prompt,
        "response": baseline_response,
    },
    {
        "system_prompt": role_system_prompt,
        "user_prompt": user_prompt,
        "response": role_response,
    },
)

# TODO: Write a constraints system prompt replacing the ***********
constraints_system_prompt = f""" {role_system_prompt}. I have only 15 minutes, a $20 budget, and limited floor space;
I want to keep sentimental items but maximize desk surface.."""

print("Sending prompt with constraints...")
constraints_response = get_completion(constraints_system_prompt, user_prompt)
print("Response received!\n")

# Show last two prompts and responses
display_responses(
    {
        "system_prompt": role_system_prompt,
        "user_prompt": user_prompt,
        "response": role_response,
    },
    {
        "system_prompt": constraints_system_prompt,
        "user_prompt": user_prompt,
        "response": constraints_response,
    },
)


reasoning_system_prompt = (
    f"{constraints_system_prompt}. Explain your reasoning for each step of the plan in a thoughtful way before presenting the final checklist."
)

print("Sending prompt with reasoning request...")
reasoning_response = get_completion(reasoning_system_prompt, user_prompt)
print("Response received!\n")

# Display the last two prompts and responses
display_responses(
    {
        "system_prompt": constraints_system_prompt,
        "user_prompt": user_prompt,
        "response": constraints_response,
    },
    {
        "system_prompt": reasoning_system_prompt,
        "user_prompt": user_prompt,
        "response": reasoning_response,
    },
)