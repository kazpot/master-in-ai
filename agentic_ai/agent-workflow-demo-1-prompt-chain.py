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

def call_openai(system_prompt, user_prompt, model="gpt-4.1-nano"):
    """Simple wrapper for OpenAI API calls"""
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

def feedstock_analyst_agent(feedstock_name):
    system_prompt = """あなたは石油化学の専門家です。
    与えられた炭化水素原料(feedstock)を分析し、主要な構成成分と、
    ガソリン・軽油・灯油などの高付加価値製品の製造への一般的な適性について
    簡潔に分析してください。
    """

    user_prompt = f"""炭化水素原料(feedstock)を分析してください: {feedstock_name}
    """

    print("石油化学の専門家が分析中")
    return call_openai(system_prompt, user_prompt)

def distillation_planner_agent(feedstock_analysis):
    system_prompt = """あなたは石油精製所の蒸留工程の専門家です。
    提供された原料分析結果（feedstock analysis）に基づいて、
    ガソリン、軽油、灯油などの主要製品の予想収率（％）を推定してください。

    現実的な推定値を提示してください。
    """

    user_prompt = f"""原料分析結果: {feedstock_analysis}
    """
    
    print("石油精製所の蒸留工程の専門家が作業中")
    return call_openai(system_prompt, user_prompt)

def market_analyst_agent(product_list):
    system_prompt = """あなたはエネルギー市場アナリストです。
    以下の精製製品について、
    * 現在の市場需要（High / Medium / Low）
    * 一般的な収益性の傾向
    を簡潔に分析してください。
    """
    
    user_prompt = f"""次の精製製品について市場分析を行ってください: {product_list}
    """

    print("エネルギー市場アナリストが分析中")
    return call_openai(system_prompt, user_prompt)

def production_optimizer_agent(distillation_plan, market_data):
    system_prompt = """あなたは石油精製所の生産最適化の専門家です。
    あなたの目的は、
    予想される製品収率と現在の市場状況を踏まえて、
    最適な生産戦略を提案することです。
    """

    user_prompt = f"""
    以下の蒸留計画があります。

    --- 蒸留計画 ---
    {distillation_plan}
    --- 蒸留計画終了 ---

    また、以下の市場分析結果があります。

    --- 市場分析 ---
    {market_data}
    --- 市場分析終了 ---

    予想される収率と市場環境の両方を考慮した上で、
    精製所が価値（収益）を最大化するために優先的に生産すべき製品、
    または重点的に取り組むべき製品について、
    簡潔に推奨事項を示してください。
    """

    print("石油精製所の生産最適化の専門家が分析中")
    return call_openai(system_prompt, user_prompt)


def run_simple_chain(current_feedstock):

    print(f"Processing feedstock: {current_feedstock}\n")

    feedstock_analysis = feedstock_analyst_agent(current_feedstock)
    print(f"\n--- Feedstock Analysis ---\n{feedstock_analysis}\n")

    product_list = distillation_planner_agent(feedstock_analysis)
    print(f"\n--- Product List ---\n{product_list}\n")

    market_data = market_analyst_agent(product_list)
    print(f"\n--- Market Analysis ---\n{market_data}\n")

    final = production_optimizer_agent(product_list, market_data)
    print(f"\n--- Optimised Production Recommendation ---\n{final}\n")

if __name__ == "__main__":
    current_feedstock = "WTI原油（ウエスト・テキサス・インターミディエイト原油）"
    results = run_simple_chain(current_feedstock)