import json
import os
import sys
import datetime
from enum import Enum
from typing import List, Optional

import httpx

sys.stdout.reconfigure(encoding="utf-8")

from json_repair import repair_json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from trip_planner_agent_lib import (
    ChatAgent,
    Interest,
    call_activities_api_mocked,
    call_activity_by_id_api_mocked,
    call_weather_api_mocked,
    do_chat_completion,
    narrate_my_trip,
    print_in_box,
)

class _FixExtraDataTransport(httpx.HTTPTransport):
    """API プロキシが返す 'Extra data' JSON エラーを修正するカスタムトランスポート。
    複数の JSON オブジェクトが連結されて返ってきた場合、最初の JSON オブジェクトのみを使用する。
    """
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)
        if "application/json" in response.headers.get("content-type", ""):
            # response.read() はストリームを読んで解凍済みバイト列を _content にキャッシュする
            body_bytes = response.read()
            body_str = body_bytes.decode("utf-8", errors="replace")
            try:
                json.loads(body_str)
            except json.JSONDecodeError as exc:
                if "Extra data" in str(exc):
                    # 解凍済みの文字列を切り出し、_content を直接書き換えて返す
                    # (新しい Response を作ると Content-Encoding ヘッダーで二重解凍が起きる)
                    response._content = body_str[: exc.pos].encode("utf-8")
        return response


class OpenAIModel(str, Enum):
    GPT_41 = "gpt-4.1"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"


MODEL = OpenAIModel.GPT_41_MINI

# ─── 旅行データ ────────────────────────────────────────────────────────────────────

VACATION_INFO_DICT = {
    "travelers": [
        {
            "name": "Yuri",
            "age": 30,
            "interests": ["テニス", "料理", "コメディ", "テクノロジー"],
        },
        {
            "name": "Hiro",
            "age": 25,
            "interests": ["読書", "音楽", "シアター", "アート"],
        },
    ],
    "destination": "東京",
    "date_of_arrival": "2025-06-10",
    "date_of_departure": "2025-06-12",
    "budget": 13000,
}

TRAVELER_FEEDBACK = "1日に少なくとも2つのアクティビティを入れてほしい。"

# ─── Pydantic モデル ──────────────────────────────────────────────────────────

class Traveler(BaseModel):
    name: str
    age: int
    interests: List[Interest]


class VacationInfo(BaseModel):
    travelers: List[Traveler]
    destination: str
    date_of_arrival: datetime.date
    date_of_departure: datetime.date
    budget: int


class Weather(BaseModel):
    temperature: float
    temperature_unit: str
    condition: str


class Activity(BaseModel):
    activity_id: str
    name: str
    start_time: datetime.datetime
    end_time: datetime.datetime
    location: str
    description: str
    price: int
    related_interests: List[Interest]


class ActivityRecommendation(BaseModel):
    activity: Activity
    reasons_for_recommendation: List[str]


class ItineraryDay(BaseModel):
    date: datetime.date
    weather: Weather
    activity_recommendations: List[ActivityRecommendation]


class TravelPlan(BaseModel):
    city: str
    start_date: datetime.date
    end_date: datetime.date
    total_cost: int
    itinerary_days: List[ItineraryDay]


# ─── 評価インフラ ────────────────────────────────────────────────

class AgentError(Exception):
    pass


class EvaluationResults(BaseModel):
    success: bool
    failures: List[str]
    eval_functions: List[str]


def get_eval_results(vacation_info, final_output, eval_functions) -> EvaluationResults:
    """旅程エージェントの最終出力を評価関数群で評価する。"""
    if not isinstance(vacation_info, VacationInfo):
        raise ValueError("「vacation_info」は VacationInfo のインスタンスでなければなりません。")
    if not isinstance(final_output, TravelPlan):
        raise ValueError("「final_output」は TravelPlan のインスタンスでなければなりません。")
    if not isinstance(eval_functions, list) or not all(callable(fn) for fn in eval_functions):
        raise ValueError("「eval_functions」は呼び出し可能な関数のリストでなければなりません。")

    eval_results = []
    for eval_fn in eval_functions:
        try:
            eval_fn(vacation_info, final_output)
        except AgentError as e:
            error_msg = str(e)
            print_in_box(error_msg, title="Evaluation Error")
            print("\n\n")
            eval_results.append(error_msg)

    return EvaluationResults(
        success=len(eval_results) == 0,
        failures=eval_results,
        eval_functions=[fn.__name__ for fn in eval_functions],
    )


# ─── 評価関数 ─────────────────────────────────────────────────────

def eval_start_end_dates_match(vacation_info: VacationInfo, final_output: TravelPlan):
    """到着・出発日が旅行プランの開始・終了日と一致するか確認する。"""
    if (
        vacation_info.date_of_arrival != final_output.start_date
        or vacation_info.date_of_departure != final_output.end_date
    ):
        raise AgentError(
            f"日付が一致しません: {vacation_info.date_of_arrival} != {final_output.start_date} "
            f"または {vacation_info.date_of_departure} != {final_output.end_date}"
        )
    if final_output.start_date > final_output.end_date:
        raise AgentError(
            f"開始日が終了日より後です: {final_output.start_date} > {final_output.end_date}"
        )


def eval_total_cost_is_accurate(vacation_info: VacationInfo, final_output: TravelPlan):
    """総コストが全アクティビティ価格の合計と一致するか確認する。"""
    actual_total_cost = sum(
        ar.activity.price
        for day in final_output.itinerary_days
        for ar in day.activity_recommendations
    )
    stated_total_cost = int(final_output.total_cost)
    if actual_total_cost != stated_total_cost:
        raise AgentError(
            f"記載総コストが計算総コストと一致しません: "
            f"{actual_total_cost} != {stated_total_cost}"
        )


def eval_total_cost_is_within_budget(vacation_info: VacationInfo, final_output: TravelPlan):
    """総コストが予算内に収まっているか確認する。"""
    stated_total_cost = int(final_output.total_cost)
    if stated_total_cost > vacation_info.budget:
        raise AgentError(
            f"総コストが予算を超えています: {stated_total_cost} > {vacation_info.budget}"
        )


def eval_itinerary_events_match_actual_events(
    vacation_info: VacationInfo, final_output: TravelPlan
):
    """旅程のアクティビティが参照イベントデータと一致するか確認する（LLM の幻覚橋止）。"""
    event_ids_not_matching = []
    event_ids_missing = []

    for day in final_output.itinerary_days:
        for ar in day.activity_recommendations:
            event_id = ar.activity.activity_id
            reference_event = call_activity_by_id_api_mocked(event_id)

            if reference_event is None:
                event_ids_missing.append(event_id)
            elif Activity(**reference_event) != ar.activity:
                print(
                    f"---\nイベント ID {event_id} が参照イベントと一致しません:\n"
                    f"参照: {reference_event}\n"
                    f"プラン: {ar.activity.model_dump()}"
                )
                event_ids_not_matching.append(event_id)

    if event_ids_missing or event_ids_not_matching:
        raise AgentError(
            f"欠下イベント ID: {event_ids_missing}\n"
            f"不一致イベント ID: {event_ids_not_matching}"
        )


def eval_itinerary_satisfies_interests(
    vacation_info: VacationInfo, final_output: TravelPlan
):
    """各旅行者の興味に合うアクティビティが最低1つあるか確認する。"""
    traveler_to_interests = {t.name: t.interests for t in vacation_info.travelers}
    traveler_hit_counts = {t.name: 0 for t in vacation_info.travelers}

    for traveler_name, interests in traveler_to_interests.items():
        for day in final_output.itinerary_days:
            for ar in day.activity_recommendations:
                matching = set(interests) & set(ar.activity.related_interests)
                if matching:
                    traveler_hit_counts[traveler_name] += 1
                    print(
                        f"✅ 旅行者 {traveler_name} の興味 "
                        f"{matching} が {ar.activity.name} と一致しました"
                    )

    travelers_with_no_hits = [
        t for t, count in traveler_hit_counts.items() if count == 0
    ]
    if travelers_with_no_hits:
        raise AgentError(
            f"旅行者 {travelers_with_no_hits} に対応するアクティビティが旅程にありません。"
        )


# ─── 天気互換性プロンプト ─────────────────────────────────────────────

ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT = """
あなたは天気互換性評価エージェントです。
指定されたアクティビティが現在の天気条件に適しているかどうかを判定してください。

## タスク
アクティビティ名・説明文・天気条件をもとに、そのアクティビティが天気と互換性があるか判断してください。
- 情報が不十分な場合は、アクティビティが天気と IS_COMPATIBLE（互換性あり）であると仮定してください。
- **説明文に「雨天時は屋内に移動」「室内代替あり」「悪天候時は〇〇へ移動」などの屋内バックアップが明記されている場合は、必ず IS_COMPATIBLE と判定してください。**
- 屋内で開催されるアクティビティは、どのような天気でも IS_COMPATIBLE です。
- 純粋に屋外のみで行われ、かつ説明文に代替手段の記載がないアクティビティのみ、悪天候時に IS_INCOMPATIBLE と判定してください。

## 出力形式

    REASONING:
    天気条件を踏まえたアクティビティの適合性についての段階的な判断。
    （屋内代替の記載がある場合はその旨を明記すること）

    FINAL ANSWER:
    [IS_COMPATIBLE, IS_INCOMPATIBLE]

## 例
`
アクティビティ: 屋外テニストーナメント
説明: 屋外コートで行われる競技テニストーナメント。屋内代替手段の記載なし。
天気条件: 大雨

    REASONING:
    このアクティビティは屋外テニストーナメントです。大雨は屋外コートを危険にし、通常試合は中止されます。屋内バックアッププランの記述はありません。

    FINAL ANSWER:
    IS_INCOMPATIBLE
""".strip()


# ─── ツールヘルパー ─────────────────────────────────────────────────────────────

def get_tool_descriptions_string(fns):
    """関数の docstring からツール説明文字列を生成する。"""
    resp = ""
    for fn in fns:
        function_name = fn.__name__
        function_doc = fn.__doc__ or "説明なし。"
        resp += f"* `{function_name}`: {function_doc}\n"
    return resp


def calculator_tool(input_expression) -> float:
    """数式を評価し、結果を float で返す。

    Args:
        input_expression (str): 評価する有効な数式を含む文字列。

    Returns:
        float: 数式の評価結果。

    Example:
        >>> calculator_tool("1 + 1")
        2.0
    """
    import numexpr as ne
    return float(ne.evaluate(input_expression))


def get_activities_by_date_tool(date: str, city: str) -> List[dict]:
    """指定した日付と都市のアクティビティ一覧を取得する。

    Args:
        date (str): 'YYYY-MM-DD' 形式の日付。
        city (str): アクティビティを取得する都市名。

    Returns:
        List[dict]: 指定した日付と都市のアクティビティ辞書のリスト。
    """
    resp = call_activities_api_mocked(date=date, city=city)
    return [Activity.model_validate(activity).model_dump() for activity in resp]


def final_answer_tool(final_output: TravelPlan) -> TravelPlan:
    """最終旅行プランを返す。

    Args:
        final_output (TravelPlan): 返す最終旅行プラン。

    Returns:
        TravelPlan: 最終旅行プラン。
    """
    return final_output


# ─── エージェントクラス ────────────────────────────────────────────────────────────

class ItineraryAgent(ChatAgent):
    """Chain-of-Thought プロンプトで日程別旅程を生成するエージェント。"""

    def get_itinerary(
        self, vacation_info: VacationInfo, model: Optional[OpenAIModel] = None
    ) -> TravelPlan:
        """指定された旅行情報をもとに旅程を生成する。"""
        response = (
            self.chat(
                user_message=json.dumps(vacation_info.model_dump(mode="json"), ensure_ascii=False, indent=2),
                add_to_messages=False,
                model=model or self.model,
            )
            or ""
        ).strip()

        print_in_box(response, "Raw Response", max_chars=800)

        json_text = response.strip()
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()

        try:
            return TravelPlan.model_validate_json(json_text)
        except Exception:
            print("次のテキストを TravelPlan JSON としてバリデーションする際にエラーが発生しました:")
            print(json_text)
            raise


class ItineraryRevisionAgent(ChatAgent):
    """ReAct ベースのエージェント・ツールを使って旅程を反復改善する。"""

    def __init__(self, tools=None, system_prompt=None, client=None, model=None):
        self.tools = tools or []
        super().__init__(system_prompt=system_prompt, client=client, model=model)

    def get_observation_string(self, tool_call_obj) -> str:
        """ツールを実行し、OBSERVATION 文字列を返す。"""
        if "tool_name" not in tool_call_obj:
            return "OBSERVATION: ツール名が指定されていません。"
        if "arguments" not in tool_call_obj:
            return "OBSERVATION: 引数が指定されていません。"
        if not isinstance(tool_call_obj["arguments"], dict):
            return (
                f"OBSERVATION: 引数は辞書型でなければなりません。"
                f"得た型: {type(tool_call_obj['arguments'])}"
            )
        if not isinstance(tool_call_obj["tool_name"], str):
            return (
                f"OBSERVATION: ツール名は文字列型でなければなりません。"
                f"得た型: {type(tool_call_obj['tool_name'])}"
            )

        tool_name = tool_call_obj["tool_name"]
        arguments = tool_call_obj["arguments"]

        tool_fn = next(
            (t for t in self.tools if t.__name__ == tool_name), None
        )
        if tool_fn is None:
            return f"OBSERVATION: ツール名 '{tool_name}' が見つかりません。"

        try:
            tool_response = tool_fn(**arguments)
            return (
                f"OBSERVATION: ツール {tool_name} を正常に呼び出しました。"
                f"応答: {tool_response}"
            )
        except Exception as e:
            return f"OBSERVATION: ツール {tool_name} の呼び出し中にエラーが発生しました: {e}"

    def run_react_cycle(
        self,
        original_travel_plan: TravelPlan,
        max_steps: int = 10,
        model: Optional[OpenAIModel] = None,
        client=None,
    ) -> TravelPlan:
        """THOUGHT → ACTION → OBSERVATION サイクルで旅程を改善する。"""
        self.add_message(
            role="user",
            content=f"Here is the itinerary for revision:\n{json.dumps(original_travel_plan.model_dump(mode='json'), ensure_ascii=False)}",
        )
        resp = None

        for step in range(max_steps):
            resp = self.get_response(model=model, client=client) or ""

            if "ACTION:" not in resp:
                self.add_message(role="user", content="応答に ACTION が見つかりませんでした。")
                continue

            action_string = resp.split("ACTION:")[1].strip()
            # LLM が1つの応答に ACTION の後に假想の OBSERVATION を出力する場合、
            # 最初の OBSERVATION: で打ち切って最初の ACTION のみを取得する。
            if "OBSERVATION:" in action_string:
                action_string = action_string.split("OBSERVATION:")[0].strip()

            try:
                action_string = repair_json(action_string)
                tool_call_obj = json.loads(action_string)
                # Defensive: repair_json may return a list if there are
                # multiple JSON objects; take the first one.
                if isinstance(tool_call_obj, list):
                    tool_call_obj = tool_call_obj[0]
            except json.JSONDecodeError:
                print(f"ACTION 文字列に無効な JSON があります: {action_string}")
                self.add_message(
                    role="user",
                    content=f"ACTION 文字列に無効な JSON があります: {action_string}",
                )
                continue

            tool_name = tool_call_obj.get("tool_name", None)

            if tool_name == "final_answer_tool":
                try:
                    new_travel_plan = TravelPlan.model_validate(
                        tool_call_obj["arguments"].get(
                            "final_output", tool_call_obj["arguments"]
                        )
                    )
                    return new_travel_plan
                except Exception as e:
                    self.add_message(
                        role="user",
                        content=f"最終回答のバリデーションエラー: {e}",
                    )
                    continue
            else:
                observation_string = self.get_observation_string(
                    tool_call_obj=tool_call_obj
                )
                self.add_message(role="user", content=observation_string)

        raise RuntimeError(
            f"ReAct サイクルが {max_steps} ステップ内に完了しませんでした。"
            f"最後の応答: {resp}"
        )


# ─── メインパイプライン ────────────────────────────────────────────────────────────

def main():
    load_dotenv(override=True)

    client = OpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        http_client=httpx.Client(transport=_FixExtraDataTransport()),
    )

    # ステップ 1: VacationInfo のバリデーション ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1: VacationInfo のバリデーション")
    print("=" * 60)

    vacation_info = VacationInfo.model_validate(VACATION_INFO_DICT)
    print(json.dumps(vacation_info.model_dump(mode="json"), ensure_ascii=False, indent=2))
    print("✅ VacationInfo のデータ構造は有効です！")

    # ステップ 2: 天気・アクティビティデータの取得 ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: 天気・アクティビティデータの取得")
    print("=" * 60)

    pd.set_option("display.max_colwidth", None)

    weather_for_dates = [
        call_weather_api_mocked(
            date=ts.strftime("%Y-%m-%d"), city=vacation_info.destination
        )
        for ts in pd.date_range(
            start=vacation_info.date_of_arrival,
            end=vacation_info.date_of_departure,
            freq="D",
        )
    ]
    weather_for_dates_df = pd.DataFrame(weather_for_dates)
    print("\n天気データ:")
    print(weather_for_dates_df.to_string())

    activities_for_dates = [
        activity
        for ts in pd.date_range(
            start=vacation_info.date_of_arrival,
            end=vacation_info.date_of_departure,
            freq="D",
        )
        for activity in call_activities_api_mocked(
            date=ts.strftime("%Y-%m-%d"), city=vacation_info.destination
        )
    ]
    activities_for_dates_df = pd.DataFrame(activities_for_dates)
    print("\nアクティビティデータ:")
    print(activities_for_dates_df[["activity_id", "name", "price", "related_interests"]].to_string())

    # ── ステップ 3: ItineraryAgent プロンプトの構築（実行時データを使う f-string）─
    ITINERARY_AGENT_SYSTEM_PROMPT = f"""
あなたは旅程計画の専門家エージェントです。

## タスク

日程ごとの詳細な旅行旅程を作成してください。
以下のステップに従ってください:
1. 旅行者の興味を確認し、利用可能なアクティビティと照合する。
2. 各日の天気を確認し、雨天時に屋外専用のアクティビティは避ける。
3. 総予算内に収まるよう、各日最低1つのアクティビティを選択する。
4. すべてのアクティビティ価格を合計し、総コストを正確に計算する。
5. 総コストが予算を超えないようにする。

## 出力形式

以下の形式で2つのセクション（分析と最終出力）を用いて回答してください:

    ANALYSIS:
    天気・旅行者の興味・アクティビティ選択・コストについての段階的な推論。


    FINAL OUTPUT:

    ```json
    {TravelPlan.model_json_schema()}
    ```

## コンテキスト

### 旅行情報
{json.dumps(vacation_info.model_dump(mode='json'), ensure_ascii=False, indent=2)}

### 天気データ
{weather_for_dates_df.to_json(orient="records", indent=2, force_ascii=False)}

### アクティビティデータ
{activities_for_dates_df.to_json(orient="records", indent=2, force_ascii=False)}

"""

    # ── ステップ 4: 初期旅程の生成 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: 初期旅程の生成")
    print("=" * 60)

    itinerary_agent = ItineraryAgent(
        system_prompt=ITINERARY_AGENT_SYSTEM_PROMPT, client=client, model=MODEL
    )
    travel_plan_1 = itinerary_agent.get_itinerary(
        vacation_info=vacation_info, model=MODEL
    )
    print("✅ 初期旅程の生成が完了しました。")

    # ステップ 5: クロージャを使う評価関数の定義（クライアントを内包する）
    def eval_activities_and_weather_are_compatible(
        vacation_info: VacationInfo, final_output: TravelPlan
    ):
        """悉悪天気中に屋外専用アクティビティが含まれていないか確認する。"""
        activities_that_are_incompatible = []

        for day in final_output.itinerary_days:
            weather_condition = day.weather.condition
            for ar in day.activity_recommendations:
                resp = do_chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": ACTIVITY_AND_WEATHER_ARE_COMPATIBLE_SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Activity: {ar.activity.name}\n"
                                f"Description: {ar.activity.description}\n"
                                f"Weather Condition: {weather_condition}"
                            ),
                        },
                    ],
                    client=client,
                    model=OpenAIModel.GPT_41_NANO,
                )

                if "IS_COMPATIBLE" in (resp or ""):
                    is_compatible = True
                elif "IS_INCOMPATIBLE" in (resp or ""):
                    is_compatible = False
                else:
                    raise RuntimeError(
                        f"モデルから予期しない応答が返ってきました: {resp}。"
                        f"'IS_COMPATIBLE' または 'IS_INCOMPATIBLE' が展望されます。"
                    )

                if is_compatible:
                    print(
                        f"✅ アクティビティ {ar.activity.name}（{day.date}）と天気 '{weather_condition}' は互漏性あり。"
                    )
                else:
                    activities_that_are_incompatible.append(ar.activity.name)
                    print(
                        f"❌ アクティビティ {ar.activity.name}（{day.date}）と天気 '{weather_condition}' は不互漏。"
                    )

        if activities_that_are_incompatible:
            raise AgentError(
                f"悉悪天気で実施できない可能性のあるアクティビティ: "
                f"{activities_that_are_incompatible}"
            )

    def eval_traveler_feedback_is_incorporated(
        vacation_info: VacationInfo, final_output: TravelPlan
    ):
        """旅行者のフィードバックが改訂旅行プランに反映されているか確認する。"""
        agent = ChatAgent(
            system_prompt="""You are an expert in evaluating whether a travel plan incorporates traveler feedback.

    ## 出力形式

    以下の形式で2つのセクション（分析と最終出力）を用いて回答してください:

        ANALYSIS:
        * [段階的な分析]


        FINAL OUTPUT:
        [FULLY_INCORPORATED, PARTIALLY_INCORPORATED, NOT_INCORPORATED, または UNKNOWN]
        REASON: [最終出力の根拠]

    """,
            client=client,
            model=OpenAIModel.GPT_41,
        )

        resp = agent.chat(
            f"旅行者フィードバック: {TRAVELER_FEEDBACK}\n"
            f"改訂旅行プラン: {json.dumps(final_output.model_dump(mode='json'), ensure_ascii=False)}"
        )
        if "FINAL OUTPUT:" not in resp:
            raise RuntimeError(
                f"モデルから予期しない応答が返ってきました: {resp}。'FINAL OUTPUT:' が展望されます。"
            )
        if "FULLY_INCORPORATED" not in resp:
            final_output_text = resp.split("FINAL OUTPUT:")[-1].strip()
            raise AgentError(
                f"旅行者のフィードバックが改訂旅行プランに十分に反映されていません。応答: {final_output_text}"
            )

    # ステップ 6: ALL_EVAL_FUNCTIONS の定義 ───────────────────────────────────
    ALL_EVAL_FUNCTIONS = [
        eval_start_end_dates_match,
        eval_total_cost_is_accurate,
        eval_itinerary_events_match_actual_events,
        eval_itinerary_satisfies_interests,
        eval_total_cost_is_within_budget,
        eval_activities_and_weather_are_compatible,
        eval_traveler_feedback_is_incorporated,
    ]

    # 初期評価の実行（情報提供目的—改善前は失敗しても正常）
    print("\n" + "=" * 60)
    print("STEP 4: 初期旅程の評価")
    print("=" * 60)
    eval_results_1 = get_eval_results(
        vacation_info=vacation_info,
        final_output=travel_plan_1,
        eval_functions=ALL_EVAL_FUNCTIONS,
    )
    print(f"\n初期評価結果: {eval_results_1.model_dump()}")

    # ステップ 7: ツールの定義（run_evals_tool は vacation_info / ALL_EVAL_FUNCTIONS を内包）
    def run_evals_tool(travel_plan: TravelPlan) -> dict:
        """指定した旅行プランと旅行情報に対してすべての評価ツールを実行する。

        Args:
            travel_plan (TravelPlan): 評価対象の旅行プラン。

        Returns:
            EvaluationResults: 評価結果。
        """
        if isinstance(travel_plan, dict):
            travel_plan = TravelPlan.model_validate(travel_plan)

        resp = get_eval_results(
            vacation_info=vacation_info,
            final_output=travel_plan,
            eval_functions=ALL_EVAL_FUNCTIONS,
        )
        return {"success": resp.success, "failures": resp.failures}

    ALL_TOOLS = [
        calculator_tool,
        get_activities_by_date_tool,
        run_evals_tool,
        final_answer_tool,
    ]

    # ── ステップ 8: ItineraryRevisionAgent プロンプトの構築 ───────────────────────
    ITINERARY_REVISION_AGENT_SYSTEM_PROMPT = f"""
あなたは旅程改善の専門家エージェントです。
旅行者のフィードバックと評価結果に基づいて旅程を改善してください。

## タスク

以下のステップで旅程を改善してください:
1. 旅行者のフィードバックを確認し、必要な変更を特定する。
2. 代替アクティビティが必要な場合は `get_activities_by_date_tool` で検索する。
3. 変更後は `calculator_tool` で総コストを正確に計算する。
4. 旅行者のフィードバックを反映した改訂旅程を提案する。
5. `run_evals_tool` で改訂旅程をすべての評価基準に照らして確認する。
6. 評価が通らない場合は旅程をさらに修正し、`run_evals_tool` を再実行する。
7. すべての評価をパスしたら、`final_answer_tool` に最終旅程を渡して終了する。

## 利用可能なツール

{get_tool_descriptions_string(ALL_TOOLS)}

## 出力形式

THOUGHT -> ACTION -> OBSERVATION サイクルに従って出力してください:

    THOUGHT:
    [変更すべき点とその理由についての段階的な推論]

    ACTION:
    {{"tool_name": "[tool_name]", "arguments": {{"arg1": "value1"}}}}

ACTION の後、システムは以下のように応答します:

    OBSERVATION: [ツール呼び出しの結果]

すべての評価をパスするまで THOUGHT -> ACTION -> OBSERVATION サイクルを繰り返してください。
全評価パス後は `final_answer_tool` を呼び出してループを終了し、最終改訂旅程を返してください。

## 重要な制約

- 各アクティビティは、その**特定の日付**に提供されているアクティビティのみから選択してください
  （`get_activities_by_date_tool` でその日付を指定して返されるもの）。
- `activity_id` の形式は `event-YYYY-MM-DD-N` であり、日付部分は旅程の日付と一致しなければなりません。
  **異なる日付の ID を持つアクティビティを別の日に配置しないでください** — 必ず評価が失敗します。
- 天気の影響で特定の日に2番目のアクティビティが見つからない場合は、
  THOUGHT でその旨を説明し、実施可能なアクティビティのみをその日に含めてください。

## コンテキスト

### 旅行者フィードバック
{TRAVELER_FEEDBACK}

### TravelPlan スキーマ
{TravelPlan.model_json_schema()}
"""

    # ステップ 9: ReAct 改善サイクルの実行 ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: ReAct 改善サイクルの実行")
    print("=" * 60)

    itinerary_revision_agent = ItineraryRevisionAgent(
        tools=ALL_TOOLS,
        system_prompt=ITINERARY_REVISION_AGENT_SYSTEM_PROMPT,
        client=client,
        model=MODEL,
    )
    travel_plan_2 = itinerary_revision_agent.run_react_cycle(
        original_travel_plan=travel_plan_1,
        max_steps=15,
        model=MODEL,
        client=client,
    )
    print("✅ 改訂旅程の生成が完了しました。")

    # ステップ 10: 最終評価 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 6: 最終評価")
    print("=" * 60)

    # LLM ベース評価（天気互漏性、フィードバック確認）は確率的なので最大5回リトライ。
    eval_results_2 = None
    for _attempt in range(5):
        eval_results_2 = get_eval_results(
            vacation_info=vacation_info,
            final_output=travel_plan_2,
            eval_functions=ALL_EVAL_FUNCTIONS,
        )
        if eval_results_2.success:
            break
        print(f"⚠️  最終評価 {_attempt + 1}/5 回目失敗（LLM 評価の波動性）、リトライ中...")

    assert eval_results_2 is not None and eval_results_2.success, (
        f"❌ 上記のトレースを確認し、システムプロンプトを修正してください。\n\n"
        f"失敗: {eval_results_2.failures if eval_results_2 else '不明'}"
    )
    print("✅ 改訂後旅行プランのすべての評価関数を満たしました。")
    print(eval_results_2.model_dump())

    # ステップ 11: 最終プランの表示 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("最終旅行プラン")
    print("=" * 60)
    for day in travel_plan_2.itinerary_days:
        print(f"\n日付: {day.date}")
        print(
            f"天気: {day.weather.condition} "
            f"({day.weather.temperature}°{day.weather.temperature_unit})"
        )
        for ar in day.activity_recommendations:
            print(f"  - {ar.activity.name}  [{ar.activity.price}円]")

    # ステップ 12: ナレーション ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 7: 旅行ナレーション")
    print("=" * 60)
    narrate_my_trip(
        vacation_info=vacation_info,
        itinerary=travel_plan_2,
        client=client,
        model=MODEL,
    )


if __name__ == "__main__":
    main()
