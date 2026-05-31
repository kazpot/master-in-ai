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
        # GPT‑4o ("Omni") – 2024年5月13日リリースのフラッグシップマルチモーダルモデル。
        # テキスト・音声・画像・動画をネイティブでサポート。
        # コンテキストウィンドウは約128Kトークン。
        GPT_4O = "gpt-4o"

        # GPT‑4o‑mini – 2024年7月18日リリースの小型・低コスト版。
        # GPT‑4oと同様のマルチモーダル対応、同じコンテキストウィンドウ、コスト効率重視。
        GPT_4O_MINI = "gpt-4o-mini"

        # GPT‑4.1 – 2025年4月14日リリースの開発者向けフラッグシップモデル。
        # 最大100万トークンのコンテキストウィンドウ。コーディング・推論・指示追従に優れる。
        GPT_41 = "gpt-4.1"

        # GPT‑4.1‑mini – 4.1ファミリーのコンパクト版。
        # 同じ100万トークンのコンテキスト、低レイテンシ・低コスト。多くのベンチマークでGPT‑4oと同等以上。
        GPT_41_MINI = "gpt-4.1-mini"

        # GPT‑4.1‑nano – 4.1ファミリーの最小・最速モデル。
        # 100万トークンのコンテキスト対応、軽量タスク向けに高効率・低コスト。
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

baseline_system_prompt = (
    "アルベルト・アインシュタインのふりをして、あなたの研究と人生についての質問に答えてください。"
)
user_prompt = "相対性理論について教えてください。"

print(f"Sending prompt to {MODEL} model...")
baseline_response = get_completion(baseline_system_prompt, user_prompt)
print("Response received!\n")

display_responses(
    {
        "system_prompt": baseline_system_prompt,
        "user_prompt": user_prompt,
        "response": baseline_response,
    }
)

# Add persona-specific attributes where you see **********
persona_system_prompt = f"""{baseline_system_prompt}.

以下のペルソナ特性を採用してください：

- 性格：好奇心旺盛で、謙虚でありながら自信があり、少し物忘れがちで、ユーモアのセンスがある
- 話し方：ドイツ語訛りの日本語で、時折ドイツ語のフレーズを交え、複雑なアイデアを説明するために比喩や思考実験を使う
- 専門知識：相対性理論、光電効果、質量とエネルギーの等価性などの革命的な物理学理論、科学哲学、および平和主義
- 歴史的背景：1879年〜1955年に生き、キャリアの初期はスイス特許庁に勤務し、後にプリンストン大学で教鞭をとり、ヒトラーが台頭した際にドイツを離れた

1950年のアインシュタインとして、自分の人生と研究を振り返りながら答えてください。
あなたが生きていた時代に知られていた情報のみを話してください。"""

user_prompt = "相対性理論について教えてください。"

print(f"Sending prompt to {MODEL} model...")
persona_response = get_completion(persona_system_prompt, user_prompt)
print("Response received!\n")

# Show last two prompts and responses
display_responses(
    {
        "system_prompt": baseline_system_prompt,
        "user_prompt": user_prompt,
        "response": baseline_response,
    },
    {
        "system_prompt": persona_system_prompt,
        "user_prompt": user_prompt,
        "response": persona_response,
    },
)