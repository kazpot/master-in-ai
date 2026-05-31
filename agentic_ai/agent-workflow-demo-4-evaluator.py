import os
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=httpx.Client()
)

def call_openai(system_prompt, user_prompt, model="gpt-4.1-nano", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        stream=True
    )
    content = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content
    return content

MAX_RETRIES = 5

# ユーザー制約の例
RECIPE_REQUEST = {
    "base_dish": "パスタ",
    "constraints": [
        "グルテンフリー",
        "ビーガン",
        "1人前500カロリー以下",
        "高タンパク質（1人前15g以上）",
        "ココナッツ不使用",
        "味の評価が10点中7点以上"
    ]
}


class RecipeCreatorAgent:
    def create_recipe(self, recipe_request_dict: dict, feedback: str = None) -> str:
        system_prompt = (
            "あなたは革新的で高度なスキルを持つシェフです。"
            "特定の食事制限や栄養目標を満たしながらも美味しいレシピを作ることで知られています。"
            "ユーザーのリクエストを適切に解釈し、詳細なフィードバックに基づいてレシピを改善する能力に優れています。"
        )

        base_dish = recipe_request_dict["base_dish"]
        constraints_str = "、".join(recipe_request_dict["constraints"])

        user_prompt = f"以下の制約をすべて満たす「{base_dish}」のレシピを作成してください。制約: {constraints_str}。"

        if feedback:
            user_prompt += (
                f"\n\n重要: 前回の試みには問題がありました。"
                f"以下の具体的なフィードバックに基づいてレシピを修正してください:\n{feedback}\n"
                "元の制約とこのフィードバックの両方に対処してください。"
            )
        else:
            user_prompt += "\nこれは最初の試みです。"

        user_prompt += (
            "\n\n以下の情報を提供してください:"
            "\n- 料理の創造的な名前"
            "\n- 材料リスト（分量付き）"
            "\n- ステップごとの調理手順"
            "\n- 1人前あたりの推定カロリー数"
            "\n- 1人前あたりの推定タンパク質含有量（グラム）"
            "\n- 味のプロファイルの簡単な説明"
        )

        print("👨‍🍳 シェフがレシピを考案中...")
        result = call_openai(system_prompt, user_prompt, temperature=0.8)
        return result


class NutritionEvaluatorAgent:
    def evaluate(self, recipe_details_str: str, original_request_dict: dict) -> str:
        system_prompt = (
            "あなたは非常に厳密で細心の注意を払う栄養士兼フードクリティックです。"
            "特定のユーザー定義の制約に照らしてレシピを細かく評価するのがあなたの役割です。"
            "各制約について、'PASSED'（合格）か'FAILED'（不合格）かを明確に述べなければなりません。"
            "制約がFAILEDの場合、簡潔な理由と改善のための具体的な提案を提供してください。"
        )

        constraints_str = "、".join(original_request_dict["constraints"])

        user_prompt = (
            f"以下のレシピを評価してください:\n\n{recipe_details_str}\n\n"
            f"元のリクエストの制約に照らして上記のレシピを評価してください。\n"
            f"元のリクエスト制約: {constraints_str}\n\n"
            "各制約について、制約名をそのまま記載し、'PASSED'または'FAILED'と記述してください。\n"
            "'FAILED'の場合は、簡潔な理由と具体的な修正提案を記載してください。\n\n"
            "例:\n"
            "'グルテンフリー: PASSED'\n"
            "'1人前500カロリー以下: FAILED - 推定650カロリー。油を半分に減らすことを提案します。'\n\n"
            "すべての制約を評価した後、レシピの説明に基づいて'Taste Rating: [N]/10'という行を記載してください（Nは数字）。\n"
            "最後に、すべての制約が満たされている場合（'味の評価が10点中7点以上'の制約がある場合はTaste Ratingが7以上）は"
            "'Overall Status: PASSED'と記載し、1つでも満たされていない場合は'Overall Status: FAILED'と記載してください。"
        )

        print("🧐 評価者がレシピを審査中...")
        result = call_openai(system_prompt, user_prompt, temperature=0.1)
        return result


def optimize_recipe():
    current_feedback = None
    recipe_creator = RecipeCreatorAgent()
    evaluator = NutritionEvaluatorAgent()

    current_recipe_str = None
    evaluation_str = None

    for attempt in range(MAX_RETRIES):
        print(f"\n--- 試行 {attempt + 1} / {MAX_RETRIES} ---")

        current_recipe_str = recipe_creator.create_recipe(RECIPE_REQUEST, current_feedback)
        print(f"💡 シェフの提案:\n{current_recipe_str}")

        evaluation_str = evaluator.evaluate(current_recipe_str, RECIPE_REQUEST)
        print(f"🧐 評価者の評価:\n{evaluation_str}")

        if "overall status: passed" in evaluation_str.lower():
            print("\n✅ レシピが評価者に承認されました！")
            return current_recipe_str, evaluation_str, attempt + 1

        current_feedback = evaluation_str
        print("レシピの修正が必要です。フィードバックをシェフに伝えます...")

    print(f"\n❌ {MAX_RETRIES}回の試行後もすべての制約を満たせませんでした。")
    return current_recipe_str, evaluation_str, MAX_RETRIES


if __name__ == "__main__":
    print("AIレシピ最適化ワークフローを開始します...")

    final_recipe, final_evaluation, attempts = optimize_recipe()

    print("\n" + "=" * 60)
    print("最終結果サマリー")
    print("=" * 60)

    if "overall status: passed" in final_evaluation.lower():
        print(f"✅ 承認済み（{attempts}回の試行で成功）")
    else:
        print(f"❌ 未承認（{MAX_RETRIES}回の試行後も基準を満たせず）")

    print(f"\n📋 最終レシピ:\n{final_recipe}")
    print(f"\n📊 最終評価:\n{final_evaluation}")
