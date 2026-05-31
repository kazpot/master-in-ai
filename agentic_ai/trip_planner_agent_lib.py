from enum import Enum

SINGLE_TAB_LEVEL = 4


class Interest(str, Enum):
    ART = "アート"
    COOKING = "料理"
    COMEDY = "コメディ"
    DANCING = "ダンス"
    FITNESS = "フィットネス"
    GARDENING = "ガーデニング"
    HIKING = "ハイキング"
    MOVIES = "映画"
    MUSIC = "音楽"
    PHOTOGRAPHY = "写真"
    READING = "読書"
    SPORTS = "スポーツ"
    TECHNOLOGY = "テクノロジー"
    THEATRE = "シアター"
    TENNIS = "テニス"
    WRITING = "ライティング"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


class ChatAgent:
    """OpenAI API を通じて会話を行うチャットエージェント。"""

    system_prompt = "あなたは親切なアシスタントです。"
    messages = []

    def __init__(self, name=None, system_prompt=None, client=None, model=None):
        self.name = name or self.__class__.__name__
        if system_prompt:
            self.system_prompt = system_prompt
        self.client = client
        self.model = model
        self.reset()

    def add_message(self, role, content):
        if role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {role}")
        self.messages.append({"role": role, "content": content})
        if role == "system":
            print_in_box(content, f"{self.name} - システムプロンプト", max_chars=150)
        elif role == "user":
            print_in_box(content, f"{self.name} - ユーザーメッセージ", max_chars=300)
        elif role == "assistant":
            print_in_box(content, f"{self.name} - アシスタント応答", max_chars=600)

    def reset(self):
        from textwrap import dedent
        system_prompt = dedent(self.system_prompt).strip()
        self.messages = []
        self.add_message("system", system_prompt)

    def get_response(self, add_to_messages=True, model=None, client=None, **kwargs):
        response = do_chat_completion(
            messages=self.messages,
            model=model or self.model,
            client=client or self.client,
            **kwargs,
        )
        if add_to_messages:
            self.add_message("assistant", response)
        return response

    def chat(self, user_message, add_to_messages=True, model=None, **kwargs):
        self.add_message("user", user_message)
        return self.get_response(add_to_messages=add_to_messages, model=model, **kwargs)


def print_in_box(text, title="", cols=120, tab_level=0, max_chars=None):
    """指定されたテキストをシンプルなヘッダー／フッター区切りで出力する。"""
    indent = "  " * tab_level
    header = f"--- {title} ---" if title else "---"
    print(f"\n{indent}{header}")
    text_str = str(text)
    if max_chars is not None and len(text_str) > max_chars:
        omitted = len(text_str) - max_chars
        text_str = text_str[:max_chars] + f"\n... （残り {omitted} 文字省略）"
    for line in text_str.splitlines():
        print(f"{indent}{line}")
    print(f"{indent}---")


def do_chat_completion(messages: list, model=None, client=None, **kwargs):
    """OpenAI チャット補完 API のシンプルなラッパー。"""
    import time

    if client is None:
        raise ValueError("有効な OpenAI クライアントを指定してください。")
    if model is None:
        raise ValueError("有効なモデルを指定してください。")

    last_exc = None
    for _attempt in range(3):
        try:
            if "response_format" not in kwargs:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
            else:
                response = client.beta.chat.completions.parse(
                    model=model,
                    messages=messages,
                    **kwargs,
                )

            if hasattr(response, "error"):
                raise RuntimeError(f"OpenAI API がエラーを返しました: {str(response.error)}")

            return response.choices[0].message.content

        except Exception as e:
            last_exc = e
            # 一時的な API エラー（JSON パースエラーなど）はリトライ
            err_str = str(e)
            if "JSONDecodeError" in type(e).__name__ or "Extra data" in err_str or "json" in err_str.lower():
                print(f"⚠️  API 一時エラー（試行 {_attempt + 1}/3）: {e}  リトライ中...")
                time.sleep(1)
                continue
            raise

    raise RuntimeError(f"do_chat_completion が3回試行後も失敗しました: {last_exc}")


ACTIVITY_CALENDAR = [
    {
        "activity_id": "event-2025-06-10-0",
        "name": "フューチャーテック朝食交流会",
        "start_time": "2025-06-10 09:00",
        "end_time": "2025-06-10 11:00",
        "location": "イノベーション・アトリウム、テクノロジー・ディストリクト、東京",
        "description": "テクノロジー愛好家のみなさんと一緒に、フューチャーテック朝食交流会で刺激的な朝を過ごしましょう！コーヒーと焼きたてのパストリーを楽しみながら、最新テクノロジーのトレンド・ガジェットデモ・ネットワーキングの機会に飛び込もう。広々としたイノベーション・アトリウムで屋内開催されるこのイベントは、アイデアを交換し、快適でモダンな空間で新たな可能性を発見したいテック愛好家にぴったりです。",
        "price": 2000,
        "related_interests": ["テクノロジー"],
    },
    {
        "activity_id": "event-2025-06-10-1",
        "name": "サーブ＆サボール：テニスと味覚のランチョン",
        "start_time": "2025-06-10 12:00",
        "end_time": "2025-06-10 13:30",
        "location": "グランド・ラケット・テラス、東京",
        "description": "東京のテニスと料理愛好家のための究極のコラボイベント「サーブ＆サボール」へようこそ！ランチタイムに屋外コートでダブルスを楽しんだ後、地元シェフが指導するハンズオン料理ワークショップでリラックスしながら、エネルギー補給できる美味しいレシピを作って楽しもう。スポーツ好きでも料理好きでも、このにぎやかな屋外ランチョンは両方の情熱を一度に楽しめます。テニスを楽しみ、料理し、新鮮な食事と楽しさを満喫したい方にぴったりです！",
        "price": 2000,
        "related_interests": ["料理", "テニス"],
    },
    {
        "activity_id": "event-2025-06-10-2",
        "name": "アートフル・アスレチックス：ペイント＆プレイ大会",
        "start_time": "2025-06-10 15:00",
        "end_time": "2025-06-10 17:00",
        "location": "クリエイティブ・コーツ・パーク、東京",
        "description": "アートとスポーツが融合するクリエイティブ・コーツ・パークでのエキサイティングな午後をご体験ください！「アートフル・アスレチックス：ペイント＆プレイ大会」では、好きなスポーツをモチーフにした屋外壁画の共同制作に参加し、楽しいスポーツミニゲームで体を動かせます。絵を描くことが好きでもプレイすることが好きでも、このイベントは青空の下でクリエイティビティ・チームワーク・動く喜びを讃えます。アート愛好家もスポーツ愛好家も大歓迎！（屋外開催；雨天時は近くのコミュニティ・ジムに移動します。）",
        "price": 1500,
        "related_interests": ["アート", "スポーツ"],
    },
    {
        "activity_id": "event-2025-06-10-3",
        "name": "東京 トワイライト・ライティング・エスケープ",
        "start_time": "2025-06-10 19:00",
        "end_time": "2025-06-10 21:00",
        "location": "インク・ロフト、12 クイル・レーン、東京",
        "description": "コーヒーと同じくらい自由に言葉が流れるインク・ロフトで、作家仲間と一緒に刺激的な夜を過ごしましょう！このライティングイベントは、小説家・詩人・ブロガー、またはストーリーテリングに情熱を持つ方すべてを歓迎します。東京の最も居心地のよいラウンジで屋内開催され、ライティングゲーム・グループプロンプト・自分の作品を読み上げる機会が盛りだくさん。このクリエイティブな屋内の場で、繋がり・創造し・ライティングの芸術を祝いましょう。",
        "price": 1500,
        "related_interests": ["ライティング", "読書", "アート"],
    },
    {
        "activity_id": "event-2025-06-11-0",
        "name": "モーニング・グルーヴ・ダンスパーティー",
        "start_time": "2025-06-11 09:00",
        "end_time": "2025-06-11 10:30",
        "location": "リズム・ホール、センター・プラザ、東京",
        "description": "モーニング・グルーヴ・ダンスパーティーでエネルギーと喜びに満ちた一日をスタートしましょう！このにぎやかなイベントは、あらゆるレベルのダンサーを歓迎し、アップビートな音楽と楽しいルーティンで活気あふれる屋内セッションを提供します。モダンポップ・ラテンビート・クラシックディスコを愛する方も、ダンスインストラクターが一緒に体を動かすサポートをしてくれます。カラフルなリズム・ホールでダンス仲間と繋がりましょう。ダンス・音楽・フィットネス好きにぴったり。リズムに乗って動こう！（屋内開催。）",
        "price": 1500,
        "related_interests": ["ダンス", "音楽", "フィットネス"],
    },
    {
        "activity_id": "event-2025-06-11-1",
        "name": "テックランチ＆ラーン：AI フロンティア",
        "start_time": "2025-06-11 12:00",
        "end_time": "2025-06-11 13:30",
        "location": "デジタル・アトリウム、東京",
        "description": "人工知能の未来を探求するダイナミックなランチタイムイベントに、テック愛好家仲間と参加しましょう！デジタル・アトリウムで屋内開催されるこのテックランチ＆ラーンでは、テクノロジーとイノベーションを軸にした短時間トーク・インタラクティブデモ・ネットワーキングの機会が充実しています。軽食を楽しみながら、テクノロジー・AI・デジタルの世界に情熱を持つ仲間と繋がろう。ベテラン開発者でもテクノロジーに興味を持ち始めたばかりでも、このイベントはあなたのためにあります！関連する興味：テクノロジー、音楽（サウンドテックデモ）、写真（AIイメージング）、ライティング（AI創造性）。",
        "price": 2000,
        "related_interests": ["テクノロジー", "音楽", "写真", "ライティング"],
    },
    {
        "activity_id": "event-2025-06-11-2",
        "name": "東京 アート＆ミュージック・フュージョン・フェスト",
        "start_time": "2025-06-11 15:00",
        "end_time": "2025-06-11 17:30",
        "location": "エコー・ガーデンズ・アンフィシアター、東京",
        "description": "アートと音楽の鮮やかな世界が融合する、エコー・ガーデンズ・アンフィシアターでの忘れられない午後に自分を浸らせましょう！緑豊かな庭園に囲まれた青空の下で、才能ある地元ミュージシャンのライブパフォーマンスを楽しみながら、東京のクリエイティブコミュニティの作品が展示されたインタラクティブな屋外アートギャラリーを探索しよう。このやりがいある屋外イベントは、感動を求め仲間のクリエイターと繋がりたいアート・音楽愛好家にぴったりです。リラックスした友好的な雰囲気の中でのメロディーと色彩の融合をお見逃しなく！",
        "price": 1800,
        "related_interests": ["アート", "音楽"],
    },
    {
        "activity_id": "event-2025-06-11-3",
        "name": "パレット＆パレート：アートと料理の融合体験",
        "start_time": "2025-06-11 18:30",
        "end_time": "2025-06-11 20:30",
        "location": "ザ・クリエイティブ・キャンバス・スタジオ、アルティザナル・レーン、東京",
        "description": "アートと料理が融合するカラフルな夜に自分を浸らせましょう！「パレット＆パレート」では、参加者がまずクリエイティブ・キャンバス・スタジオで料理にインスパイアされた傑作を描く屋内ガイドセッションに参加し、その後地元シェフが率いる料理クラスで鮮やかな食べられるアート作品の作り方を学びます。アート愛好家でも料理好きでもどちらでも、このクリエイティブな夜は色と風味を通じた自己表現と交流にぴったりです！全素材・食材が提供されます。このイベントは屋内開催で、アートと料理のあらゆる経験レベルを歓迎します。",
        "price": 2500,
        "related_interests": ["アート", "料理"],
    },
    {
        "activity_id": "event-2025-06-12-0",
        "name": "東京 ネイチャー＆グリーンサム・アドベンチャー",
        "start_time": "2025-06-12 08:00",
        "end_time": "2025-06-12 10:00",
        "location": "エコー・リッジ植物トレイル、東京",
        "description": "ハイキングとガーデニングを融合した屋外アドベンチャーの朝を、自然愛好家仲間と楽しみましょう！絵のようなエコー・リッジのトレイルを穏やかにハイキングしながら、専門ガイドが地元の植物生態を紹介し、実践的なガーデニングのコツを教えてくれます。ミニ植え付けで手を土で汚し、在来種の育て方を学びましょう。ハイキングとガーデニングどちらも好きな方にぴったりな、新鮮な空気・コミュニティ・緑のインスピレーションに満ちた屋外イベントです。",
        "price": 1500,
        "related_interests": ["ハイキング", "ガーデニング"],
    },
    {
        "activity_id": "event-2025-06-12-1",
        "name": "サウンドトラック・ピクニック：映画と音楽のランチタイム",
        "start_time": "2025-06-12 12:00",
        "end_time": "2025-06-12 13:30",
        "location": "スターライト・アンフィシアター、東京",
        "description": "屋外スターライト・アンフィシアターで、クラシック映画のシーンとライブ音楽の魔法を体験しましょう！ランチを持参して芝生でくつろぎながら、ミュージシャンが映画の名曲を演奏し、選ばれたシーンがオープンスクリーンに映し出されます。映画ファンも音楽ファンも大歓迎のこのイベントは、晴れたランチタイムに両アートを讃えます。雨天時はイベントが隣接するハーモニー・ホールに移動します。音楽のために来て、映画のシネマティックな感動のために残ろう！",
        "price": 1500,
        "related_interests": ["映画", "音楽"],
    },
    {
        "activity_id": "event-2025-06-12-2",
        "name": "トレイル・テールズ：ライティング＆ハイキング・アドベンチャー",
        "start_time": "2025-06-12 14:00",
        "end_time": "2025-06-12 16:30",
        "location": "ウィスパリング・パインズ・トレイルヘッド、東京",
        "description": "東京の景色豊かなトレイルで、仲間の愛好家と共に屋外ライティングの旅に出発しましょう！「トレイル・テールズ」は、美しい松林を歩くハイキングと自然にインスパイアされた創作ライティングを組み合わせたユニークなイベントです。詩・物語・日記のどれが好きでも、ライティングとハイキングどちらも楽しみたい方にぴったりです。ガイド付きプロンプト・共同作業・たっぷりの新鮮な空気が用意されています。野外を探索しながら創造力を養いたいあらゆるレベルのライターに最適です。",
        "price": 2000,
        "related_interests": ["ライティング", "ハイキング"],
    },
    {
        "activity_id": "event-2025-06-12-3",
        "name": "テック＆フィルム・フュージョン・ナイト",
        "start_time": "2025-06-12 19:00",
        "end_time": "2025-06-12 21:30",
        "location": "バーチャルリアリティー・シアター、シリコン・プラザ、東京",
        "description": "映画の魔法と最新テクノロジーが出会う没入型の夜に飛び込もう！映画ファンとテック愛好家が集い、最先端のSF短編映画の特別上映と、地元映画制作者およびVRテクノロジストとのインタラクティブなパネルディスカッションを楽しみます。エンターテインメントの未来を体験し、テクノロジーが映画の世界をどう変えているかを語り合おう。バーチャルリアリティー・シアターで開催される屋内のこのエキサイティングなイベントは、テクノロジーと映画に興味を持つすべての人におすすめです。",
        "price": 1500,
        "related_interests": ["テクノロジー", "映画"],
    },
    {
        "activity_id": "event-2025-06-13-0",
        "name": "ラフ＆グルーヴ：朝のコメディ・ダンス・バッシュ",
        "start_time": "2025-06-13 09:00",
        "end_time": "2025-06-13 10:30",
        "location": "ジャイビング・パーラー、セントラル・プラザ、東京",
        "description": "東京中心部のジャイビング・パーラーで開催される屋内イベント「ラフ＆グルーヴ朝のコメディ・ダンス・バッシュ」で、大笑いと大きなムーブで一日をスタートしましょう！このにぎやかなイベントは面白いスタンドアップパフォーマンスとアップビートなグループダンスセッションを融合します。ダンスが好きでも、コメディが好きでも、ただ楽しく一日を始めたいだけでも、ここに居場所があります。ダンスとコメディ両方のファンにぴったり—笑って、踊って、繋がる準備をしてきてください！",
        "price": 1500,
        "related_interests": ["ダンス", "コメディ"],
    },
    {
        "activity_id": "event-2025-06-13-1",
        "name": "トレイルズ＆テールズ：ランチタイム・ハイキング＆ライティング・リトリート",
        "start_time": "2025-06-13 12:00",
        "end_time": "2025-06-13 13:30",
        "location": "ウィスパリング・パインズ・トレイルヘッド、東京",
        "description": "東京の美しい森で、清々しいハイキングと創作ライティングを組み合わせたユニークなランチタイムイベント「トレイルズ＆テールズ」で大自然に飛び込もう！ガイド付きハイキングで景色を楽しみながら、自然愛好のライター仲間と一緒に景観のよい場所で立ち止まり、反省や執筆に時間を使おう。ハイキング愛好家でも、ライティングに情熱を持っていても、自然の中でクリエイティブに充電したくても、この屋外アドベンチャーはあなたのためにあります！注意：このイベントは屋外開催です。筆記用具と軽食が提供されます。",
        "price": 1500,
        "related_interests": ["ハイキング", "ライティング"],
    },
    {
        "activity_id": "event-2025-06-13-2",
        "name": "アート＆レンズ：屋外クリエイティブ・ウォーク",
        "start_time": "2025-06-13 15:00",
        "end_time": "2025-06-13 17:00",
        "location": "サンセット・プロムナード・アートパーク、東京",
        "description": "アート愛好家と写真愛好家が集う「アート＆レンズ：屋外クリエイティブ・ウォーク」に参加しましょう！東京のサンセット・プロムナード・アートパークの鮮やかな景色を探索しながら、インスピレーションを感じる瞬間をカメラに収めてスケッチしよう。カメラ・スケッチブック、またはその両方を持参して、繋がり・学び・創造する機会が豊富なガイド付きクリエイティブな旅をお楽しみください。このやりがいあるイベントは完全屋外で開催され、アートと写真に情熱を持つすべての方に最適です。",
        "price": 1500,
        "related_interests": ["アート", "写真"],
    },
    {
        "activity_id": "event-2025-06-13-3",
        "name": "サンセット・グルーヴ・ハイク",
        "start_time": "2025-06-13 18:00",
        "end_time": "2025-06-13 20:00",
        "location": "スターリット・リッジ、東京",
        "description": "サンセット・グルーヴ・ハイクで冒険とリズムの完璧な融合を体験しましょう！ハイキングとダンス愛好家の仲間と共に、夕日が沈む中でスターリット・リッジの景色豊かなトレイルを歩きます。活力あふれるハイキングの途中で、パノラマビューポイントで立ち止まり、プロのインストラクター主導のグループダンスセッションを楽しもう。この屋外イベントはハイキングの喜びとダンスの楽しさを融合し、自然の中での忘れられない夕べを提供します。すべての経験レベルを歓迎—青空の下で動いて踊ろう！",
        "price": 1500,
        "related_interests": ["ハイキング", "ダンス"],
    },
    {
        "activity_id": "event-2025-06-14-0",
        "name": "サンライズ・ネイチャー＆プラント・ウォーク",
        "start_time": "2025-06-14 08:00",
        "end_time": "2025-06-14 10:00",
        "location": "エメラルド・メドウズ・パーク、東京",
        "description": "ハイキングとガーデニングの完璧な融合を体験しましょう！景色豊かなエメラルド・メドウズ・パークを爽快に散策する朝のハイキングに参加しながら、道中で地元の植物について学び、ハンズオンのガーデニング活動をしましょう。ハイキング愛好家でも植物ファンでも、この屋外アドベンチャーはあなたのためにあります。自然や仲間の愛好家と繋がりながら、心身と地元の植物生態を育てよう。",
        "price": 1500,
        "related_interests": ["ハイキング", "ガーデニング"],
    },
    {
        "activity_id": "event-2025-06-14-1",
        "name": "ランチタイム・ブルーム：コミュニティ・ガーデン・パーティー",
        "start_time": "2025-06-14 12:00",
        "end_time": "2025-06-14 13:30",
        "location": "グリーン・ヘイブン・パーク、東京",
        "description": "東京の中心で開催される活気あふれる屋外ガーデニングイベントに参加しましょう！「ランチタイム・ブルーム」は植物ファンや自然愛好家に最適な集まりです。実践的なガーデニングのコツを学び、花を植えるセッションに参加し、同じ志を持つグリーンサム仲間と繋がろう。ガーデニングの世界を探索しながら新鮮な空気の中で軽食を楽しもう。経験豊富なガーデナーでも始めたばかりでも、この活気あふれる屋外イベントはすべての年齢に向けたインスピレーションと楽しさをお届けします。",
        "price": 1500,
        "related_interests": ["ガーデニング", "フィットネス"],
    },
    {
        "activity_id": "event-2025-06-14-2",
        "name": "東京 サマー・ガーデン・パーティー",
        "start_time": "2025-06-14 14:00",
        "end_time": "2025-06-14 16:30",
        "location": "ブルーミング・コートヤード、東京",
        "description": "東京サマー・ガーデン・パーティーで花咲く午後の楽しさを体験しましょう！経験豊富なガーデナーでも始めたばかりでも、この屋外イベントはみんなが一緒に掘り起こして育てることを楽しめます。ハンズオンワークショップを探索し、持ち帰れる花を植え、ガーデンゲームを楽しみ、同じ植物愛好家仲間と繋がろう。このイベントはガーデニングと自然に興味を持つすべての方に教育・クリエイティビティ・リラクゼーションを組み合わせてお届けします。屋外・緑を愛する家族・友人・ソロ冒険家にぴったりです！",
        "price": 1500,
        "related_interests": ["ガーデニング", "アート", "フィットネス"],
    },
    {
        "activity_id": "event-2025-06-14-3",
        "name": "ダンシング・スルー・プローズ：クリエイティブなムーブメント＆ライティングの夕べ",
        "start_time": "2025-06-14 19:00",
        "end_time": "2025-06-14 21:00",
        "location": "ライターズ・ワルツ・ホール、東京",
        "description": "ダンスとライティングの世界が美しく交差する活気あふれる夜「ダンシング・スルー・プローズ」に参加しましょう！地元の振付家と創作ライターによる指導のもと、動きからインスピレーションを受けて言葉を綴り、また散文のリズムがダンスフロアへと誘います。ダンスとライティングどちらが好きでも、経験を問わずこのイベントに参加できます。魅力的なライターズ・ワルツ・ホールで屋内開催され、想像力が本当に動き出せる活気あふれてサポーティブな雰囲気を楽しめます。このユニークな動きとストーリーテリングの祝典をお見逃しなく！",
        "price": 1500,
        "related_interests": ["ダンス", "ライティング"],
    },
    {
        "activity_id": "event-2025-06-15-0",
        "name": "ライターズ・サンライズ・ワークショップ",
        "start_time": "2025-06-15 09:00",
        "end_time": "2025-06-15 11:00",
        "location": "スターライト・リテラリー・カフェ、東京",
        "description": "スターライト・リテラリー・カフェで朝のクリエイティビティをスタートさせましょう！同じ志を持つライター仲間と一緒に、居心地のよい屋内カフェの雰囲気に囲まれたインスピレーションあふれるライティングワークショップに参加しよう。小説に取り組んでいても、詩を探求していても、日記をつけていても、このイベントは同じ熱意を持つ愛好家と繋がるのに最適です。ライティングプロンプト・グループディスカッション・たっぷりのコーヒーをお楽しみください。ライティング・読書・アートに興味を持つすべての方に最適です。（屋内開催）",
        "price": 1500,
        "related_interests": ["ライティング", "読書", "アート"],
    },
    {
        "activity_id": "event-2025-06-15-1",
        "name": "ランチタイム・グルーヴ：東京 ダンス・ソーシャル",
        "start_time": "2025-06-15 12:00",
        "end_time": "2025-06-15 13:30",
        "location": "サンビーム・コミュニティ・ホール、東京",
        "description": "ランチタイム・グルーヴ（東京の活気あふれる屋内ダンス・ソーシャル）に参加しましょう！1時間半のエネルギッシュなダンス・楽しい振付・素晴らしい音楽で昼の憂鬱を吹き飛ばそう。初心者でも経験者でも、新しい動きを学んでダンス仲間と出会うのを楽しんでください。関連する興味：ダンス、音楽、フィットネス。",
        "price": 1500,
        "related_interests": ["ダンス", "音楽", "フィットネス"],
    },
    {
        "activity_id": "event-2025-06-15-2",
        "name": "東京 サマー・ダンス・ジャム",
        "start_time": "2025-06-15 15:00",
        "end_time": "2025-06-15 17:00",
        "location": "グルーヴ・パビリオン、セントラルパーク、東京",
        "description": "東京のサマー・ダンス・ジャムで午後いっぱい踊りましょう！ダンス好きでも踊ってみたいだけでも、セントラルパークの広々としたオープンエアのグルーヴ・パビリオンに集まろう。ポップ・サルサ・スウィングの軽快なミックス曲、インタラクティブなグループレッスン、楽しいダンスオフが待っています。この屋外イベントはすべてのレベルと年齢のダンサーを歓迎します。新しい友達を作り、新しい動きを学び、音楽とダンスへの愛を讃えよう！",
        "price": 1500,
        "related_interests": ["ダンス", "音楽", "フィットネス"],
    },
    {
        "activity_id": "event-2025-06-15-3",
        "name": "トワイライト・テニス・ラリー",
        "start_time": "2025-06-15 18:00",
        "end_time": "2025-06-15 20:00",
        "location": "グランド・コーツ（サンフィールド・パーク）、東京",
        "description": "グランド・コーツで夕日が沈む中、テニスのスリリングな屋外夜間イベントに参加しましょう！初心者でも経験者でも、このイベントではテニス愛好家仲間との友好的な試合・スキルチャレンジ・交流の機会を提供します。テニスへの愛をテーマにした新鮮な空気・にぎやかな音楽・エキサイティングなプレゼントを楽しもう。ラリーしてサーブして楽しもう！テニス・フィットネス・新しい人との出会いに情熱を持つ方にぴったりです。",
        "price": 1500,
        "related_interests": ["テニス", "フィットネス", "音楽"],
    },
]

INCLIMATE_WEATHER_CONDITIONS = ["雷雨", "雨"]

WEATHER_FORECAST = [
    {
        "date": "2025-06-10",
        "city": "東京",
        "temperature": 27,
        "temperature_unit": "摂氏",
        "condition": "快晴",
        "description": "東京は快晴で青空が広がり、気温も高め。屋外アクティビティに最高の一日です！",
    },
    {
        "date": "2025-06-11",
        "city": "東京",
        "temperature": 29,
        "temperature_unit": "摂氏",
        "condition": "曇り時々晴れ",
        "description": "晴れ間と雲が混在する温かな一日。屋外活動を楽しむのに絶好のコンディションです。",
    },
    {
        "date": "2025-06-12",
        "city": "東京",
        "temperature": 26,
        "temperature_unit": "摂氏",
        "condition": "雷雨",
        "description": "午後から雷雨の予報。強い雨と突風が予想され、湿度も高く蒸し暑い一日となりそうです。空には次第に厚い雲が広がり、劇的な雰囲気になるでしょう。",
    },
    {
        "date": "2025-06-13",
        "city": "東京",
        "temperature": 20,
        "temperature_unit": "摂氏",
        "condition": "雨",
        "description": "一日を通して断続的な雨が降り、肌寒い風が吹く曇り空。時折激しい雷雨になる可能性もあります。",
    },
    {
        "date": "2025-06-14",
        "city": "東京",
        "temperature": 18,
        "temperature_unit": "摂氏",
        "condition": "雨",
        "description": "終日しとしとと雨が降り続き、曇り空で気温も低め。路面が滑りやすくなるため傘の携帯をおすすめします。",
    },
    {
        "date": "2025-06-15",
        "city": "東京",
        "temperature": 28,
        "temperature_unit": "摂氏",
        "condition": "晴れ",
        "description": "雨の心配がない晴天で、屋外アクティビティに最適な一日です。",
    },
]


def call_activities_api_mocked(
    date: str | None = None,
    city: str | None = None,
    activity_ids: list | None = None,
) -> list:
    """モックアクティビティ API を呼び出し、指定した日付・都市のアクティビティ一覧を返す。"""
    import datetime

    if city and city != "東京":
        return []

    if date:
        try:
            datetime.datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            print(f"日付フォーマットが無効です: {date}")
            return []

    if date and (date < "2025-06-10" or date > "2025-06-15"):
        print(f"日付 {date} は有効範囲外です（2025-06-10 〜 2025-06-15）")
        return []

    activities = ACTIVITY_CALENDAR

    if date:
        activities = [event for event in activities if event["start_time"].startswith(date)]

    if activity_ids:
        activities = [event for event in activities if event["activity_id"] in activity_ids]

    if not activities:
        print(f"{city} の {date} にアクティビティが見つかりませんでした。")
    return activities


def call_activity_by_id_api_mocked(activity_id: str):
    """モックアクティビティ API を呼び出し、ID でアクティビティを取得する。"""
    for event in ACTIVITY_CALENDAR:
        if event["activity_id"] == activity_id:
            return event
    print(f"ID {activity_id} のイベントが見つかりませんでした。")
    return None


def call_weather_api_mocked(date: str, city: str) -> dict:
    """指定した日付・都市の天気予報を返す。"""
    import datetime

    if city != "東京":
        return {}

    try:
        datetime.datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        print(f"日付フォーマットが無効です: {date}")
        return {}

    if date < "2025-06-10" or date > "2025-06-15":
        print(f"日付 {date} は有効範囲外です（2025-06-10 〜 2025-06-15）")
        return {}

    return next(
        (forecast for forecast in WEATHER_FORECAST if forecast["date"] == date), {}
    )


def narrate_my_trip(vacation_info, itinerary, client, model, filename="/tmp/my_trip_narration.mp3"):
    """旅行のナレーション要約を生成し、オプションで音声ファイルを出力する。"""
    resp = do_chat_completion(
        messages=[
            {
                "role": "user",
                "content": f"""
                以下はオンボーディングエージェントが収集した旅行情報です:
                {vacation_info}。

                以下が最終旅程です:
                {itinerary}

                旅行者・興味・制約・総費用を紹介したうえで、旅程の各日について説明してください。

                各アクティビティの個別費用は記載しないでください。

                ナレーション自体については言及しないでください。
                """,
            }
        ],
        client=client,
        model=model,
    )

    # Display the narrative (works in both notebook and script environments)
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, display
        shell = get_ipython()
        # kernel 属性があるときのみ Jupyter 環境とみなす
        if shell is not None and hasattr(shell, "kernel"):
            display(Markdown(resp))
        else:
            raise RuntimeError("Jupyter 環境ではありません")
    except Exception:
        print(resp)

    # Attempt to generate and play audio (optional, non-critical)
    try:
        if resp:
            with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice="coral",
                input=resp,
                instructions="Speak in a cheerful and positive tone.",
            ) as response:
                response.stream_to_file(filename)

            try:
                from IPython.display import Audio, display
                display(Audio(filename))
            except Exception:
                print(f"Audio saved to: {filename}")
    except Exception:
        pass
