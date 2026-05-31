import json as _json
import os
import re
import ast
import operator
from dotenv import load_dotenv

load_dotenv(override=True)

import httpx
import pandas as pd
from IPython.display import Markdown, display
from lib import (
    OpenAIModels,
    print_in_box,
    get_competitor_pricing_data,
    get_completion,
    get_promotions_data,
    get_sales_data,
    get_weather_data,
    call_weather_api
)
from openai import OpenAI

MODEL = OpenAIModels.GPT_41_NANO

class _FixVocareumTransport(httpx.HTTPTransport):
    """Vocareum API が末尾に余分なJSONを付加するバグのワークアラウンド。"""
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)
        body = response.read()
        text = body.decode("utf-8")
        try:
            _json.loads(text)
        except _json.JSONDecodeError:
            decoder = _json.JSONDecoder()
            try:
                _, end = decoder.raw_decode(text)
                text = text[:end]  # slice as string to handle multi-byte chars
            except Exception:
                pass
        # response.read() returns already-decoded bytes; strip encoding headers
        # to prevent httpx from trying to decompress again
        headers = [
            (k, v) for k, v in response.headers.raw
            if k.lower() not in (b"content-encoding", b"transfer-encoding")
        ]
        return httpx.Response(
            status_code=response.status_code,
            headers=headers,
            content=text.encode("utf-8"),
            request=request,
        )


# If using the Vocareum API endpoint
# No changes needed in this cell
# TODO: Fill in the missing parts marked with **********

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=httpx.Client(transport=_FixVocareumTransport()),
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

sales_data = get_sales_data()
sales_df = pd.DataFrame(sales_data)

promotions_data = get_promotions_data()
promotions_df = pd.DataFrame(promotions_data)

weather_data = get_weather_data()
weather_df = pd.DataFrame(weather_data)

competitor_pricing_data = get_competitor_pricing_data()
competitor_pricing_df = pd.DataFrame(competitor_pricing_data)

react_system_prompt = """
あなたはツール呼び出しと推論を使って、多段階のプロセスでどんなTASKも解決できる、細心の注意を払う小売需要アナリストです。

## 手順:
- 以下の方法でステップバイステップの推論を行います：
    - 最終的な答えに一歩近づくために、次に取るべきステップと次のツール呼び出しをTHINKする
    - 次に取るべき単一のツール呼び出しをACTする
- 常に以下のフォーマットで単一のTHINK/ACTメッセージとして応答します：
    THINK:
    [ツール呼び出しを必要としない推論を行う]
    [どのデータが必要でどのツールが利用可能かに基づいて、次のツール呼び出しについての結論]
    ACT:
    [使用するツールと引数]
- 最終的な答えが分かったら、`ACT`メッセージ内で`final_answer`ツールを呼び出してください。
- ACT:の後に必ずツール呼び出しを提供してください。そうしないと失敗します。

## 利用可能なツール
* `calculator(expression: str)`: 算術計算を実行する
    - 例:
        - 入力: `ACT: calculator(expression="(10 + 20) / 2.0")`
        - 出力: `OBSERVE: 15.0`
* `get_sales_data()`: 販売データを取得する
    - 例:
        - 入力: `ACT: get_sales_data()`
        - 出力: `OBSERVE: {"date": "2024-01-10", "product_id": "P001", "product_name": "Product 1", "quantity": 255, "revenue": 15547.35}`
* `call_weather_api(date: str)`: 指定日の気象データを取得する。急増した各日付に対して呼び出すこと。
    - 例:
        - 入力: `ACT: call_weather_api(date="2024-01-10")`
        - 出力: `OBSERVE: {"date": "2024-01-10", "weather": "Sunny", "temperature": 72}`
* `final_answer(amount_after_spike: str, causes: list[str], date: str, percentage_spike: str)`: 最終回答を返す
    - 例:
        - 入力: `ACT: final_answer(amount_after_spike="32", causes=["競合他社Xが29%割引を提供しカテゴリーへの関心が高まった", ...], date="2020-06-12", percentage_spike="20.00%")`
        - 出力: `OBSERVE: {"amount_after_spike": "32", "causes": [...], "date": "2020-06-12", "percentage_spike": "20.00%"}`

他のツールは使用しません。

例:

```
--ユーザーメッセージ--
TASK:
「1週間前の天気は何でしたか？」というクエリに答えてください。今日は2024-01-17です。

--アシスタントメッセージ--
THINK:
* 2024-01-17から1週間前の日付を計算する必要があります。
* 今日が2024-01-17であれば、7日前は2024-01-10です。
* `call_weather_api`ツールを呼び出して、2024-01-10の天気データを取得できます。
* 天気データが取得できたら、`final_answer`ツールを使って最終的な答えを返せます。
* 必要なツール呼び出し：2024-01-10で`call_weather_api`ツールを呼び出す。
ACT:
call_weather_api(date="2024-01-10")

--ユーザーメッセージ--
OBSERVE:
{"date": "2024-01-10", "weather": "Sunny"}

--アシスタントメッセージ--
THINK:
* 2024-01-10の天気データを取得しました。
* `final_answer`ツールを使って最終的な答えを返せます。
* 必要なツール呼び出し：天気データを使って`final_answer`ツールを呼び出す。
ACT:
final_answer("2024-01-10の天気は晴れでした。")

--ユーザーメッセージ--
OBSERVE:
2024-01-10の天気は晴れでした。
```
"""

user_prompt_analyze = """
TASK: 天気などの要因に基づいて、増加率で見た最大の売上急増を1件特定し、その理由を簡潔に説明してください。
"""

print(f"Sending prompt to {MODEL} model...")

messages = []
messages.append({"role": "system", "content": react_system_prompt})
messages.append({"role": "user", "content": user_prompt_analyze})

react_response = get_completion(messages=messages, model=MODEL, client=client)

messages.append({"role": "assistant", "content": react_response})
print("Response received!\n")


for message in messages:
    if message["role"] == "system":
        continue
    print_in_box(message["content"], title=f"{message['role'].capitalize()}")

assert "ACT:" in messages[-1]["content"], (
    " ❌ No ACT message found in response. Looking for: \n\n ACT:"
)


def safe_eval(expr):
    """
    Evaluate a mathematical expression safely.

    We normally don't want to use eval() because it can execute arbitrary code, unless we are in a
    properly sandboxed environment. This function is a safe alternative for evaluating mathematical
    expressions.
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](eval_node(node.operand))
        elif isinstance(node, ast.Expr):
            return eval_node(node.value)
        else:
            raise TypeError(f"Unsupported type: {type(node)}")

    result = eval_node(ast.parse(expr, mode="eval").body)

    if isinstance(result, float):
        return round(result, 2)
    elif isinstance(result, int):
        return result
    else:
        raise RuntimeError(f"Unsupported result type: {type(result)}")


def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression safely.
    """
    return float(safe_eval(expression))  # TODO: Replace with a call to evaluate the expression


assert (actual := calculator("10 + 10")) == 20.0, f" ❌ Expected 20.0, got {actual}"

def get_observation_message(response: str) -> str:
    """
    Take a THINK/ACT response, run the tool call, and return the observation message.

    Args:
        response (str): The THINK/ACT response.

    Returns:
        str: The observation message.

    Uses regular expressions to match the tool call and run the corresponding tool.

    If the response is invalid, return an error message as a string that the agent can understand.
    """
    from ast import literal_eval

    observation_message = None

    SALES_DATA_REGEX = r"ACT:\n(?:#[^\n]*\n)*get_sales_data\(\)"
    WEATHER_REGEX = r"ACT:\n(?:#[^\n]*\n)*call_weather_api\(date=\"(.*)\"\)"
    CALCULATOR_REGEX = r"ACT:\n(?:#[^\n]*\n)*calculator\(expression=\"(.*)\"\)"
    FINAL_ANSWER_REGEX = r"ACT:\n(?:#[^\n]*\n)*final_answer\(amount_after_spike=\"(.*)\", causes=(.*), date=\"(.*)\", percentage_spike=\"(.*)\"\)"

    # TOOL 1: get_sales_data
    if re.search(SALES_DATA_REGEX, response):
        sales_data = get_sales_data(products=["P005"])
        # filter sales data to Product 5
        sales_data = [
            item for item in sales_data if item["product_name"] == "Product 5"
        ]
        observation_message = f"OBSERVE:\n{sales_data}"

    # TOOL 2: call_weather_api
    elif re.search(WEATHER_REGEX, response):
        date = re.search(WEATHER_REGEX, response).groups()[0]
        weather_data = call_weather_api(date)
        observation_message = f"OBSERVE:\n{weather_data}"

    # TOOL 3: calculator
    elif re.search(CALCULATOR_REGEX, response):
        expression = re.search(CALCULATOR_REGEX, response).groups()[0]
        observation_message = f"OBSERVE:\n{calculator(expression)}"

    # TOOL 4: final_answer
    elif re.search(FINAL_ANSWER_REGEX, response):
        amount_after_spike, causes, date, percentage_spike = re.search(
            FINAL_ANSWER_REGEX,
            response,
        ).groups()
        causes = literal_eval(causes)
        observation_message = f"OBSERVE:\namount_after_spike: {amount_after_spike}\ndate: {date}\npercentage_spike: {percentage_spike}\ncauses: {causes}"

    # Error
    else:
        observation_message = "OBSERVE:\nInvalid tool call or tool not supported."

    return observation_message


# Test cases
assert (
    actual := get_observation_message("""
THINK:
[thinking here]
ACT:
get_sales_data()
""")
) == (expected := "OBSERVE:\n" + str(get_sales_data(products=["P005"]))), (
    f"{actual} != {expected}"
)

assert (
    actual := get_observation_message("""
THINK:
[thinking here]
ACT:
call_weather_api(date="2024-01-12")
""")
) == (expected := "OBSERVE:\n" + str(call_weather_api("2024-01-12"))), (
    f"{actual} != {expected}"
)

assert (
    actual := get_observation_message("""
THINK:
[thinking here]
ACT:
final_answer(amount_after_spike="10", causes=["cause1", "cause2"], date="2024-01-12", percentage_spike="10%")
""")
) == (
    expected
    := "OBSERVE:\namount_after_spike: 10\ndate: 2024-01-12\npercentage_spike: 10%\ncauses: ['cause1', 'cause2']"
), f"{actual} != {expected}"

assert (
    actual := get_observation_message("""
THINK:
[thinking here]
ACT:
calculator(expression="10 + 10")
""")
) == (expected := "OBSERVE:\n20.0"), f"{actual} != {expected}"

assert (
    actual := get_observation_message("""
THINK:
[thinking here]
ACT:
invalid_tool()
""")
) == (expected := "OBSERVE:\nInvalid tool call or tool not supported."), (
    f"{actual} != {expected}"
)

assert (
    actual := get_observation_message("""
THINK:
[thinking here]
ACT_TYPO:
get_sales_data()
""")
) == (expected := "OBSERVE:\nInvalid tool call or tool not supported."), (
    f"{actual} != {expected}"
)


# ReACT loop!
messages = []

messages.append({"role": "system", "content": react_system_prompt})
messages.append({"role": "user", "content": user_prompt_analyze})


for message in messages:
    if message["role"] == "system":
        continue
    print_in_box(message["content"], title=f"{message['role'].capitalize()}")

num_react_steps = 0

observation_message = None
while True:
    react_response = get_completion(messages=messages, model=MODEL, client=client)
    messages.append({"role": "assistant", "content": react_response})

    print_in_box(
        react_response, title=f"Assistant (Think + Act). Step {num_react_steps + 1}"
    )

    observation_message = get_observation_message(react_response)
    messages.append({"role": "user", "content": observation_message})

    if "ACT:\nfinal_answer" in react_response:
        print_in_box(observation_message, title="FINAL ANSWER")
        break

    print_in_box(
        observation_message, title=f"User (Observe). Step {num_react_steps + 1}"
    )

    num_react_steps += 1
    if num_react_steps > 10:
        print("ERROR: Max number of React steps exceeded. Breaking.")
        break

assert "date: 2024-01-12" in observation_message, "ReACT Loop did not find the spike date"
assert "percentage_spike: 200" in observation_message, "ReACT Loop did not find the spike percentage increase"