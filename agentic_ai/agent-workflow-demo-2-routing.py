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

def product_researcher_agent(query):
    """商品調査エージェント。商品に関する情報を収集する。"""

    system_prompt = """
    あなたは小売企業向けの商品調査エージェントです。

    あなたの役割は、商品に関する構造化された情報、
    市場動向、および競合他社の価格情報を調査・整理することです。
    """

    user_prompt = f"""
    次の商品について詳細に調査してください。

    {query}
    """

    return call_openai(system_prompt, user_prompt)

def customer_analyzer_agent(query):
    """顧客分析エージェント。顧客データやフィードバックを分析する。"""

    system_prompt = """
    あなたは顧客分析エージェントです。

    あなたの役割は、顧客のフィードバック、
    嗜好、購買パターンを分析することです。
    """

    user_prompt = f"""
    次の商品について顧客行動を分析してください。

    {query}
    """

    return call_openai(system_prompt, user_prompt)

def pricing_strategist_agent(query, product_data=None, customer_data=None):
    system_prompt = """あなたは小売価格戦略の専門家です。
    あなたの目的は、詳細な商品調査と顧客分析の結果に基づいて、
    最適な価格戦略と適切な販売価格を提案することです。
    """
    
    user_prompt = f"""
    元の価格設定に関する問い合わせ:
    {query}

    商品調査データ:
    {product_data}

    顧客分析データ:
    {customer_data}
    
    上記すべての情報に基づいて、
    推奨される価格戦略、最適な価格または価格帯、
    およびその判断理由を提示してください。
    """

    return call_openai(system_prompt, user_prompt)


# --- Routing Agent with LLM-Based Task Determination ---
def routing_agent(query, context=None):
    system_prompt = """
    あなたは小売業向けの問い合わせを適切なエージェントへ振り分けるAIアシスタントです。

    ユーザーからクエリが与えられるので、
    そのタスクを担当すべきエージェントを判断してください。

    利用可能なエージェント:

    - 商品調査エージェント（Product Researcher Agent）
      商品仕様、市場動向、競合価格を調査する。

    - 顧客分析エージェント（Customer Analyzer Agent）
      顧客のフィードバック、嗜好、購買パターンを分析する。

    - 価格戦略エージェント（Pricing Strategist Agent）
      調査結果と分析結果に基づき、最適な価格戦略を提案する。

    回答はエージェント名のみを返してください。
    その他の説明は不要です。
    """
    
    user_prompt = f"""
    次の問い合わせについて、どのエージェントが対応すべきですか？

    '{query}'
    """
    
    agent_choice = call_openai(system_prompt, user_prompt)
    print(f"選択されたエージェント: {agent_choice}")

    agent_choice_lower = agent_choice.lower()

    # Route the query to the correct agent based on the choice
    if "product researcher" in agent_choice_lower or "商品調査" in agent_choice:
        print("商品調査エージェントへルーティング中...")
        return product_researcher_agent(query)

    elif "customer analyzer" in agent_choice_lower or "顧客分析" in agent_choice:
        print("顧客分析エージェントへルーティング中...")
        return customer_analyzer_agent(query)

    elif "pricing strategist" in agent_choice_lower or "price strategist" in agent_choice_lower or "価格戦略" in agent_choice:
        print("価格戦略エージェントへルーティング中...")
        
        product_data = None
        if context and "product_data" in context:
            product_data = context["product_data"]
        else:
            print("商品情報を取得中...")
            product_data = product_researcher_agent(query)
        
        customer_data = None
        if context and "customer_data" in context:
            customer_data = context["customer_data"]
        else:
            print("カスタマーインサイトを取得中...")
            customer_data = customer_analyzer_agent(query)
        
        return pricing_strategist_agent(query, product_data, customer_data)
    
    else:
        return f"Couldn't route query. Agent decision was: {agent_choice}"


if __name__ == "__main__":
    queries = [
        "ワイヤレスイヤホンの仕様と現在の市場動向を調査してください。",
        "当社のプレミアムコーヒーブランドについて顧客はどのように評価していますか？",
        "新しいオーガニックスキンケア商品の最適な販売価格はいくらにすべきですか？"
        ]
    

    for query in queries:
        print(f"\nクエリ: {query}")
        
        result = routing_agent(query)
        print("\n結果:")
        print(result)
        print("\n" + "-"*80)