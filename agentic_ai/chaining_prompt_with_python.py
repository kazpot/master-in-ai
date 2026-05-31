import os
from openai import OpenAI
import httpx
from dotenv import load_dotenv

from lib import (
    OpenAIModels,
    get_completion,
)

load_dotenv(override=True)

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=httpx.Client()
)

MODEL = OpenAIModels.GPT_41_NANO

# --- Step 1: Generate the Outline ---

prompt_step1 = """
You are a helpful programming assistant.

I need a Python script to read a CSV file named 'input_data.csv',
calculate the average of a column named 'value', and write the
average to a new file named 'output.txt'.

Please provide a simple, step-by-step outline for this script.
"""

print("--- Calling AI for Step 1: Outline Generation ---")
outline_response = get_completion(user_prompt=prompt_step1, model=MODEL, client=client)

print("\n--- AI-Generated Outline ---")
print(outline_response)

# --- Step 2: Generate the Code ---

# The second prompt, which USES the output from the first
prompt_step2 = f"""
You are a helpful programming assistant.

Based on the following outline, please write the complete Python code for the script.
Ensure you use standard libraries and include comments.

Outline:
---
{outline_response}
---
"""

print("\n--- Calling AI for Step 2: Code Generation ---")
code_response = get_completion(user_prompt=prompt_step2, model=MODEL, client=client)

print("\n--- AI-Generated Python Code ---")
print(code_response)

# --- Step 3: Gate Check ---
# Gate Check（ゲートチェック）とは、AIが生成したコードを次のステップに渡す前に
# 「品質の門番」として検証を行う処理のこと。
# ここでは以下の2段階で検証している：
#   1. AIの返答からMarkdownコードフェンス(```python ... ```)を除去して純粋なPythonコードを取り出す
#   2. Pythonの標準ライブラリ ast.parse() でコードを構文解析し、構文エラーがないか確認する
# Gate Checkをパスしたコードだけを後続の処理（実行・保存など）に進めることで、
# 明らかに壊れたコードが次のエージェントステップに流れるのを防ぐ。

import ast
import re

def extract_code(text):
    """Strips Markdown code fences (```python ... ```) from AI responses."""
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def check_python_syntax(code):
    """Checks for syntax errors in a string of Python code."""
    try:
        ast.parse(code)
        return True, "No syntax errors found."
    except SyntaxError as e:
        return False, f"Syntax Error: {e}"

# Extract pure Python code before validation
extracted_code = extract_code(code_response)
is_valid, message = check_python_syntax(extracted_code)

print(f"\n--- Gate Check Result ---")
print(f"Code is valid: {is_valid}")
print(message)
