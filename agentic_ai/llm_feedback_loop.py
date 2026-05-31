import traceback
import io
import os
from contextlib import redirect_stdout, redirect_stderr
from enum import Enum
from pprint import pprint

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# セットアップ
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=httpx.Client(),
)


class OpenAIModels(str, Enum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"


MODEL = OpenAIModels.GPT_41_NANO


def get_completion(messages=None, system_prompt=None, user_prompt=None, model=MODEL):
    messages = list(messages)
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return response.choices[0].message.content


def execute_code(code, test_cases):
    results = {"execution_error": None, "test_results": [], "passed": 0, "failed": 0}
    namespace = {}
    output_buffer = io.StringIO()

    try:
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            exec(code, namespace)

        for i, test in enumerate(test_cases):
            inputs = test["inputs"]
            expected = test["expected"]
            try:
                if isinstance(inputs, dict):
                    actual = namespace["process_data"](**inputs)
                else:
                    actual = namespace["process_data"](*inputs)

                passed = actual == expected
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

                results["test_results"].append(
                    {
                        "test_id": i + 1,
                        "inputs": inputs,
                        "expected": expected,
                        "actual": actual,
                        "passed": passed,
                    }
                )
            except Exception as e:
                passed = isinstance(expected, type) and isinstance(e, expected)
                results["test_results"].append(
                    {
                        "test_id": i + 1,
                        "inputs": inputs,
                        "expected": expected,
                        "error": str(e),
                        "passed": passed,
                    }
                )
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

    except Exception as e:
        results["execution_error"] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }

    results["stdout"] = output_buffer.getvalue()
    return results


def format_feedback(results):
    feedback = []

    if results["execution_error"]:
        feedback.append(
            f"エラー: コードの実行が失敗しました ({results['execution_error']['error_type']})"
        )
        feedback.append(f"メッセージ: {results['execution_error']['error_message']}")
        feedback.append("トレースバック:")
        feedback.append(results["execution_error"]["traceback"])
        feedback.append("\n構文エラーまたは実行時エラーを修正してください。")
        return "\n".join(feedback)

    feedback.append(
        f"テスト結果: {results['passed']} 件成功, {results['failed']} 件失敗"
    )

    if results["stdout"]:
        feedback.append(f"\n標準出力:\n{results['stdout']}")

    if results["failed"] > 0:
        feedback.append("\n失敗したテストケース:")
        for test in results["test_results"]:
            if not test.get("passed"):
                feedback.append(f"\nテスト #{test['test_id']}:")
                feedback.append(f"  入力: {test['inputs']}")
                feedback.append(f"  期待値: {test['expected']}")
                if "actual" in test:
                    feedback.append(f"  実際の値: {test['actual']}")
                if "error" in test:
                    feedback.append(f"  エラー: {test['error']}")

    return "\n".join(feedback)


def extract_code(code):
    lines = code.split("\n")
    start = lines.index("```python") + 1
    end = lines.index("```", start)
    return "\n".join(lines[start:end])


# ---------------------------------------------------------------------------
# Task description
# ---------------------------------------------------------------------------

task_description = """
`process_data` という名前のPython関数を作成します。
この関数は数値データを解析するもので、以下の要件を満たす必要があります。

1. 数値のリストと、オプションのパラメータ 'mode' を受け取ること。
   'mode' には 'sum'（合計）または 'average'（平均）を指定でき、デフォルトは 'average' とする。
2. mode が 'sum' の場合、すべての数値の合計を返す。
3. mode が 'average' の場合、すべての数値の平均（mean）を返す。

使用例:
    process_data([1, 2, 3, 4, 5], mode='average')  # 戻り値: 3.0
    process_data([1, 2, 'a', 3], mode='sum')  # 戻り値: 6
"""

# ---------------------------------------------------------------------------
# 拡張テストケース（中央値・非数値の除外・空リストの処理を含む）
# ---------------------------------------------------------------------------

test_cases = [
    {"inputs": ([1, 2, 3, 4, 5], "sum"), "expected": 15},
    {"inputs": ([1, 2, 3, 4, 5], "average"), "expected": 3.0},
    {"inputs": ([11, 12, 13, 14, 15], "sum"), "expected": 65},
    {"inputs": ([11, 12, 13, 14, 15], "average"), "expected": 13.0},
    {"inputs": ([], "sum"), "expected": None},
    {"inputs": ([1, 3, 4], "median"), "expected": 3},
    {"inputs": ([1, 2, 3, 5], "median"), "expected": 2.5},
    {"inputs": ([1, 2, "a", 3], "sum"), "expected": 6},
    {"inputs": ([1, 2, None, 3, "b", 4], "average"), "expected": 2.5},
    {"inputs": ([10], "median"), "expected": 10},
    {"inputs": ([], "median"), "expected": None},
    {"inputs": ([1, 2, 3, 4, 5], "invalid_mode"), "expected": ValueError},
]

# ---------------------------------------------------------------------------
# メイン処理: フィードバックループ
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    initial_prompt = f"""
あなたは優秀なPython開発者です。

{task_description}

関数のコードのみを ```python と ``` で囲んで出力してください。説明や使用例は不要です。

出力形式の例:

```python
def process_data(data, mode='average'):
    # ここに実装
    pass
```
"""

    iterations = []

    # 初回コード生成
    messages = [{"role": "user", "content": initial_prompt}]
    initial_response = get_completion(messages)
    initial_code = extract_code(initial_response)

    initial_results = execute_code(initial_code, test_cases)
    initial_feedback = format_feedback(initial_results)

    print("【初回生成コード】")
    print(initial_code)
    print("\n【テスト結果】")
    print(initial_feedback)

    iterations.append(
        {
            "iteration": 0,
            "code": initial_code,
            "test_results": {
                "passed": initial_results["passed"],
                "failed": initial_results["failed"],
            },
        }
    )
    pprint(iterations[-1]["test_results"])

    current_code = initial_code
    current_feedback = initial_feedback

    # フィードバックループ（最大10回）
    for i in range(10):
        if iterations[-1]["test_results"]["failed"] == 0:
            print("\n成功！すべてのテストが通過しました。")
            break

        feedback_prompt = f"""
    あなたは優秀なPython開発者です。以下の要件に基づいて関数を実装しました。

    {task_description}

    現在の実装は以下の通りです:
    ```python
    {current_code}
    ```
    テストを実施した結果は以下の通りです:
    {current_feedback}
    問題を修正し、すべてのテストケースをパスするようにコードを改善してください。
    改善した関数のコードのみを出力し、説明は不要です。
    """

        messages = [{"role": "user", "content": feedback_prompt}]
        improved_response = get_completion(messages)
        improved_code = extract_code(improved_response)

        improved_results = execute_code(improved_code, test_cases)
        improved_feedback = format_feedback(improved_results)

        iterations.append(
            {
                "iteration": i + 1,
                "code": improved_code,
                "test_results": {
                    "passed": improved_results["passed"],
                    "failed": improved_results["failed"],
                },
            }
        )
        pprint(iterations[-1]["test_results"])

        current_code = improved_code
        current_feedback = improved_feedback

    # 最終サマリー
    print("\n--- イテレーションサマリー ---")
    pprint(
        [{"iteration": it["iteration"], "test_results": it["test_results"]} for it in iterations],
        width=200,
    )

    print("\n--- 最終コード ---")
    print(iterations[-1]["code"])
