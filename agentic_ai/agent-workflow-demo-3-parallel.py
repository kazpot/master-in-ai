import os
import threading
import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=httpx.Client()
)

def call_openai(system_prompt, user_prompt, model="gpt-4.1-nano"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )
    content = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            content += chunk.choices[0].delta.content
    return content

# エージェントの出力をスレッドセーフに収集するための共有辞書
agent_outputs = {}

contract_text = """
コンサルティング契約書

本コンサルティング契約書（以下「本契約」）は、2025年1月1日（以下「発効日」）を有効日として、デラウェア州法人ABCコーポレーション（以下「クライアント」）と、カリフォルニア州有限責任会社XYZコンサルティングLLC（以下「コンサルタント」）との間で締結される。

1. 業務内容。コンサルタントはクライアントに対し、経営戦略コンサルティング、市場分析、およびテクノロジー導入に関するアドバイス（以下「本業務」）を提供するものとする。

2. 契約期間。本契約は発効日に開始し、特段の事由による早期終了がない限り、12ヶ月間継続するものとする。

3. 報酬。クライアントはコンサルタントに対し、本業務の対価として月額10,000ドルを支払うものとする。支払いはコンサルタントの請求書受領後30日以内に行うものとする。

4. 機密保持。コンサルタントは、業務遂行中に機密情報にアクセスする可能性があることを認識し、当該情報の機密性を維持することに同意する。

5. 知的財産権。コンサルタントが開発したすべての成果物はクライアントの財産とする。コンサルタントは当該成果物に関するすべての権利、権原および利益をクライアントに譲渡する。

6. 契約解除。いずれの当事者も、30日前の書面による通知をもって本契約を解除できる。クライアントは解除日までに実施された業務に対する報酬を支払うものとする。

7. 準拠法。本契約はデラウェア州法に準拠するものとする。

8. 責任の制限。コンサルタントの責任は、本契約に基づきクライアントが支払った報酬額を上限とする。

9. 補償。クライアントはコンサルタントに対し、クライアントが提供した資料の使用に起因するすべての請求から生じる損害を補償するものとする。

10. 完全合意。本契約は当事者間の完全な合意を構成し、従前のすべての合意に優先するものとする。

上記の証として、当事者は上記に記載された日付をもって本契約を締結した。
"""

class LegalTermsChecker:
    """契約書内の問題のある法的条件・条項をチェックするエージェント。"""
    def run(self, contract_text):
        print("LegalTermsChecker: 法的条項を分析中...")
        system_prompt = (
            "あなたは契約法を専門とする法律の専門家です。"
            "提供された契約書を精査し、問題のある条項、曖昧な用語、非標準的な法的文言を特定してください。"
            "主要な問題点をリスト形式で報告してください。"
        )
        agent_outputs["legal"] = call_openai(system_prompt, contract_text)
        print("LegalTermsChecker: 分析完了")

class ComplianceValidator:
    """契約書の規制・業界コンプライアンスを検証するエージェント。"""
    def run(self, contract_text):
        print("ComplianceValidator: コンプライアンスを検証中...")
        system_prompt = (
            "あなたは規制コンプライアンスの専門家です。"
            "提供された契約書を審査し、一般的なビジネス規制や業界標準（データプライバシー、労働法、知的財産法など）への準拠状況を評価してください。"
            "コンプライアンス上の懸念点や不足している標準条項をリスト形式で報告してください。"
        )
        agent_outputs["compliance"] = call_openai(system_prompt, contract_text)
        print("ComplianceValidator: 検証完了")

class FinancialRiskAssessor:
    """契約書の財務リスクと負債を評価するエージェント。"""
    def run(self, contract_text):
        print("FinancialRiskAssessor: 財務リスクを評価中...")
        system_prompt = (
            "あなたは契約財務リスク分析の専門家です。"
            "提供された契約書を分析し、財務的なリスクや潜在的な負債（支払い条件、責任制限、補償義務、ペナルティなど）を評価してください。"
            "財務リスクをリスト形式で報告してください。"
        )
        agent_outputs["financial"] = call_openai(system_prompt, contract_text)
        print("FinancialRiskAssessor: 評価完了")

class SummaryAgent:
    """各専門エージェントの結果を統合するエージェント。"""
    def run(self, contract_text, inputs):
        print("SummaryAgent: 全分析結果を統合中...")
        legal_findings = inputs.get("legal", "法的分析の結果がありません。")
        compliance_findings = inputs.get("compliance", "コンプライアンス分析の結果がありません。")
        financial_findings = inputs.get("financial", "財務リスク分析の結果がありません。")

        system_prompt = (
            "あなたはシニア法律顧問です。"
            "法的条項、コンプライアンス、財務リスクの各専門家から受け取った契約書分析を統合し、"
            "経営幹部向けの包括的なエグゼクティブサマリーを作成してください。"
            "全体的なリスク評価と主要な懸念事項を明確に示してください。"
        )
        user_prompt = f"""以下の契約書分析を統合し、包括的なサマリーレポートを作成してください。
        
        【契約書テキスト（参考）】
        {contract_text[:500]}...
        
        【法的条項分析】
        {legal_findings}
        
        【コンプライアンス検証】
        {compliance_findings}
        
        【財務リスク評価】
        {financial_findings}
        
        上記を踏まえ、主要な問題点と全体的な評価を含む統合エグゼクティブサマリーを提供してください。
        """

        result = call_openai(system_prompt, user_prompt)
        print("SummaryAgent: 統合完了")
        return result

# すべてのエージェントを並列実行するメイン関数
def analyze_contract(contract_text):
    """すべてのエージェントを並列実行し、結果をまとめる。"""
    legal_checker = LegalTermsChecker()
    compliance_validator = ComplianceValidator()
    financial_assessor = FinancialRiskAssessor()
    summary_creator = SummaryAgent()

    threads = [
        threading.Thread(target=legal_checker.run, args=(contract_text,)),
        threading.Thread(target=compliance_validator.run, args=(contract_text,)),
        threading.Thread(target=financial_assessor.run, args=(contract_text,)),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final_report = summary_creator.run(contract_text, agent_outputs)
    return final_report

if __name__ == "__main__":
    print("エンタープライズ契約書分析システム")
    print("契約書を分析中...")
    final_analysis = analyze_contract(contract_text)
    print("\n=== 契約書分析結果 ===\n")
    print(final_analysis)