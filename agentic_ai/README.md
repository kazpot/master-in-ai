# agentic_ai

OpenAI API を活用した Agentic AI のサンプルコード集です。

## セットアップ

### 1. 環境変数の設定

`.env.example` をコピーして `.env` を作成し、APIキーを設定します。

```bash
cp .env.example .env
```

`.env` を編集して、APIキーとエンドポイントURLを設定してください。

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
```

### 2. Python ライブラリのインストール

```bash
pip install -r requirements.txt
```

#### conda 環境を使う場合

```bash
conda create -n agentic_ai python=3.11
conda activate agentic_ai
pip install -r requirements.txt
```

## ファイル構成

| ファイル                                       | 説明                                                 |
| ---------------------------------------------- | ---------------------------------------------------- |
| `lib.py`                                       | 共通ユーティリティ（`get_completion` など）          |
| `openai_test.py`                               | OpenAI API の基本接続テスト                          |
| `role-based-prompt.py`                         | ロールベースプロンプトのサンプル                     |
| `role-based-prompt-einstein.py`                | アインシュタインキャラクターを使ったロールプロンプト |
| `chaining_prompt_with_python.py`               | プロンプトチェーニングのサンプル                     |
| `react-prompting.py`                           | ReAct プロンプティングのサンプル                     |
| `lesson-3-prompt-instruction-refinement.ipynb` | プロンプト改善に関する Jupyter Notebook              |

## 実行例

```bash
python openai_test.py
python chaining_prompt_with_python.py
```
