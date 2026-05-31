import os
import re
import httpx
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

DEFAULT_SYSTEM_PROMPT = "あなたは役に立つアシスタントです。"

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

def extract_xml(text: str, tag: str) -> str:
    """XMLスタイルタグ間のコンテンツを抽出する。"""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def parse_tasks(xml: str) -> List[Dict]:
    """<task> XMLブロックを辞書にパースする。"""
    tasks = []
    current_task = {}

    for line in xml.splitlines():
        line = line.strip()
        if line.startswith("<task>"):
            current_task = {}
        elif line.startswith("<type>"):
            current_task["type"] = line[6:-7].strip()
        elif line.startswith("<description>"):
            current_task["description"] = line[13:-14].strip()
        elif line.startswith("</task>"):
            if "description" in current_task:
                if "type" not in current_task:
                    current_task["type"] = "default"
                tasks.append(current_task)
    return tasks

# === ワーカーエージェント基底クラス ===

class WorkerAgent:
    """全ての特化型ワーカーエージェントの抽象基底クラス。"""
    def __init__(self, task_type: str):
        self.task_type = task_type

    def run(self, original_task: str, task_description: str) -> str:
        raise NotImplementedError("'run' メソッドはサブクラスで実装する必要があります。")

# === ワーカーエージェント実装 ===

class HematologyAgent(WorkerAgent):
    """血球数（全血球計算）を分析するワーカー。"""
    def run(self, original_task: str, task_description: str) -> str:
        prompt = f"""
        あなたは血液学分析の専門家です。検査レポートの血球数セクションを解釈するのが役割です。

        メインタスク: {original_task}
        あなたのサブタスク: {task_description}

        以下の点について回答し、必ず <response>...</response> タグで囲んで返してください：
        - これらの血液値を分析する目的を説明してください。
        - 範囲外の値（例：高い/低い RBC、WBC、血小板）を特定してください。
        - 異常値の潜在的な臨床的意義を簡潔に記述してください。
        """
        raw_output = call_openai(DEFAULT_SYSTEM_PROMPT, prompt)
        return extract_xml(raw_output, "response") or raw_output

class RenalFunctionAgent(WorkerAgent):
    """腎機能マーカーを分析するワーカー。"""
    def run(self, original_task: str, task_description: str) -> str:
        prompt = f"""
        あなたは腎機能分析の専門家です。検査レポートから腎臓関連マーカーを解釈するのが役割です。

        メインタスク: {original_task}
        あなたのサブタスク: {task_description}

        以下の点について回答し、必ず <response>...</response> タグで囲んで返してください：
        - これらの腎臓マーカーを分析する目的を説明してください。
        - 範囲外の値（例：クレアチニン、BUN、GFR）を特定してください。
        - 異常値の潜在的な臨床的意義を簡潔に記述してください。
        """
        raw_output = call_openai(DEFAULT_SYSTEM_PROMPT, prompt)
        return extract_xml(raw_output, "response") or raw_output

class LiverFunctionAgent(WorkerAgent):
    """肝酵素マーカーを分析するワーカー。"""
    def run(self, original_task: str, task_description: str) -> str:
        prompt = f"""
        あなたは肝機能分析の専門家です。検査レポートの肝酵素セクションを解釈するのが役割です。

        メインタスク: {original_task}
        あなたのサブタスク: {task_description}

        以下の点について回答し、必ず <response>...</response> タグで囲んで返してください：
        - これらの肝酵素を分析する目的を説明してください。
        - 範囲外の値（例：ALT、AST、ALP）を特定してください。
        - 異常値の潜在的な臨床的意義を簡潔に記述してください。
        """
        raw_output = call_openai(DEFAULT_SYSTEM_PROMPT, prompt)
        return extract_xml(raw_output, "response") or raw_output

# === オーケストレーター ===

class Orchestrator:
    def __init__(self, orchestrator_prompt: str):
        self.orchestrator_prompt = orchestrator_prompt

    ############################################################################
    ##                                                                        ##
    ##          [  チャレンジ：ワーカーディスパッチャーを実装せよ ]              ##
    ##                                                                        ##
    ##  このメソッドを実装してください。受け取った `task_type`（"hematology"    ##
    ##  や "renal" などの文字列）を見て、正しい特化型ワーカーエージェントの      ##
    ##  インスタンスを返す必要があります。                                       ##
    ##                                                                        ##
    ############################################################################
    def get_worker(self, task_type: str) -> WorkerAgent:
        """タスクタイプを検査し、正しい特化型エージェントを返す。"""
        type_lower = task_type.lower()

        if type_lower == "hematology":
            return HematologyAgent(task_type)
        elif type_lower == "renal":
            return RenalFunctionAgent(task_type)
        elif type_lower == "liver":
            return LiverFunctionAgent(task_type)


        # 一致するものがない場合、エラーを発生させるのがベストプラクティスです。
        raise ValueError(f"タスクタイプ '{task_type}' に対応するワーカーエージェントが設定されていません。")

    def process(self, task: str) -> Dict:
        """オーケストレーター〜ワーカーのワークフロー全体を実行する。"""
        orchestrator_input = self.orchestrator_prompt.format(task=task)
        response = call_openai(DEFAULT_SYSTEM_PROMPT, orchestrator_input)
        print("\n[オーケストレーター生出力]\n", response)

        analysis = extract_xml(response, "analysis")
        tasks_xml = extract_xml(response, "tasks")
        tasks = parse_tasks(tasks_xml)

        print("\n=== オーケストレーター分析 & プラン ===")
        print("分析:", analysis)
        print("パース済みタスク:", tasks)

        results = []
        for task_info in tasks:
            try:
                agent = self.get_worker(task_info["type"])
                result = agent.run(task, task_info["description"])
                print(f"\n=== {task_info['type'].upper()} 結果 ===\n{result}")
                results.append({
                    "type": task_info["type"],
                    "description": task_info["description"],
                    "result": result
                })
            except ValueError as e:
                print(f"\n--- エラー --- \n{e}")

        return {"analysis": analysis, "worker_results": results}

# === オーケストレーター用プロンプトテンプレート ===

orchestrator_prompt = """
あなたは臨床検査データアナリストです。患者の検査結果セットを分析し、体系的に解釈するためのプランを作成するのが役割です。

プランは、検査レポートの主要なパネルごとにサブタスクに分解する必要があります。

以下のフォーマットで、<analysis>セクションと<tasks>セクションを含む形で返答してください。

<analysis>
存在する検査パネルの概要と、解釈の全体的な目標を高レベルでまとめてください。
</analysis>

<tasks>
データの中で見つかった主要な検査パネルごとに<task>エントリを1つ記述してください。各タスクには<type>と<description>が必要です。
タスクフォーマットの例：
<task>
  <type>hematology</type>
  <description>RBC、WBC、血小板を含む全血球計算（CBC）パネルを分析する。</description>
</task>
</tasks>

以下が高レベルのタスクとデータです：
タスク: {task}
"""

# === メインランナー ===

if __name__ == "__main__":
    lab_results_data = """
    患者検査レポート：
    - パネル: 全血球計算（CBC）
      - 白血球数（WBC）: 11.5 x10^9/L（正常値: 4.5-11.0）
      - 赤血球数（RBC）: 4.6 x10^12/L（正常値: 4.2-5.4）
      - 血小板数: 140 x10^9/L（正常値: 150-450）
    - パネル: 腎機能パネル
      - クレアチニン: 1.4 mg/dL（正常値: 0.6-1.2）
      - BUN: 25 mg/dL（正常値: 7-20）
    - パネル: 肝機能パネル
      - ALT: 55 U/L（正常値: 7-56）
      - AST: 60 U/L（正常値: 10-40）
    """

    user_prompt = f"以下の検査結果を解釈し、サマリーを提供してください: {lab_results_data}"

    orchestrator = Orchestrator(orchestrator_prompt)
    final_report = orchestrator.process(user_prompt)

    print("\n\n=== 最終解釈レポート ===")
    print("総合分析:\n", final_report.get("analysis", "N/A"))
    for r in final_report.get("worker_results", []):
        print(f"\n--- {r['type'].upper()} パネル ---")
        print("タスク説明:", r["description"])
        print("解釈:\n", r["result"])
