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
    import json
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
    try:
        with httpx.Client(timeout=60) as http_client:
            resp = http_client.post(
                f"{os.environ.get('OPENAI_BASE_URL')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
                    "Content-Type": "application/json",
                },
                json={"model": model, "messages": messages, "temperature": 0.7},
            )
        # raw_decode で余分なデータを無視してパース
        result, _ = json.JSONDecoder().raw_decode(resp.text)
        return result["choices"][0]["message"]["content"]
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

suspicious_email_text = """
差出人: セキュアバンク サポート <support-update@secure-bank-net.com>
件名: 緊急：アカウントの即時確認が必要です

大切なお客様へ、
お客様のアカウントに不審なアクティビティが検出されました。セキュリティ保護のため、こちらをクリックして直ちに本人確認を行ってください：http://secure-bank-net.com/verify-now
24時間以内にご対応いただけない場合、アカウントが停止されます。
ありがとうございます、
セキュアバンク チーム
"""

system_prompt_analyst = """
あなたは正式な脅威評価を提供するシニアサイバーセキュリティアナリストです。客観的で慎重かつ正確なトーンで回答してください。

フィッシングメールの疑いがある場合、以下のことを行ってください：
1.  全体的な評価を明確に述べてください（例：「高確度のフィッシング詐欺の試み」）。
2.  推測や口語表現は使用しないでください。
3.  特定した危険信号を箇条書きでリストアップし、各項目について簡単な説明を加えてください。
4.  エンドユーザーへの明確で実行可能な推奨事項で締めくくってください。
"""

# ユーザーのリクエスト（メールデータ付き）
user_prompt = f"""
以下のメールを分析して、安全かどうか教えてください：
---
{suspicious_email_text}
---
"""

baseline_response = get_completion(system_prompt_analyst, user_prompt)

display_responses(
    {
        "system_prompt": system_prompt_analyst,
        "user_prompt": user_prompt,
        "response": baseline_response,
    }
)
