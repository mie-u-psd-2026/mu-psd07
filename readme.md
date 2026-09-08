# 生成AI活用サンプルアプリ

# 概要

このアプリは Python とVue.jsを用いて作られた簡易的な生成AI活用アプリです。

- フロントエンドに、Vue.js CDN版を用いています。

- バックエンドに、Python,FlaskとOpenAI APIを用いてローカル起動のOllamaを叩いています。

# フォルダ構成

```text
frontend/
  index.html          # Vue.jsによる画面・入力・通信
backend/
  app.py              # Flask API、モデル設定、画面の配信
  prompt.txt          # 日本語の質問・結果を生成する指示
  requirements.txt    # Pythonの依存ライブラリ
  test_app.py         # APIと画面配信の自動テスト
```

相談用AIは `gemma3:4b` を使用します。モデルの詳細は [Ollama公式ページ](https://ollama.com/library/gemma3) を参照してください。
変更する場合は `backend/app.py` の既定値、または環境変数 `OLLAMA_MODEL` で指定します。
初回はモデルの読み込みに時間がかかることがあります。

# 環境
- Vscode
- OpenCode
- ollama

# 開発ツールインストール

- 管理者権限でコマンドプロンプトを起動します。

- 以下のコマンドを実行し、必要なソフトウェアを入手します。

```
winget install --id Microsoft.VisualStudioCode -e --source winget --accept-package-agreements --accept-source-agreements
winget install --id Python.Python.3.13 -e --source winget --accept-package-agreements --accept-source-agreements
winget install --id SST.opencode -e --source winget --accept-package-agreements --accept-source-agreements
winget install --id Ollama.Ollama -e --source winget --accept-package-agreements --accept-source-agreements
start /b ollama serve > NUL 2>&1
timeout /t 3 /nobreak > NUL
ollama pull gemma3:4b
```

- vscodeを起動し、アクティビティバーの拡張機能から、以下のプラグインをインストールしてください。
  - Python
  - Vue.js Extension Pack

# 環境セットアップ

- Python ライブラリインストール

  以下のコマンドでPythonの利用ライブラリをインストールします。

  ```
  pip install -r backend/requirements.txt
  ```

# 実行方法

- プロジェクトのルートで以下のコマンドを実行します。Ollamaも起動しておいてください。

  ```
  python backend/app.py
  ```

- ブラウザで以下のURLにアクセスしてみてください。

  ```
  http://localhost:5000
  ```

- フロントエンドはFlaskが配信するため、別のサーバーは不要です。
- 自動テスト：`python -m unittest discover -s backend -p "test_*.py"`
- `/send_api` には `{"consultation":"相談内容","history":[]}` をPOSTします。履歴は `{"question":"質問","answer":"回答"}` の配列です。応答は仕様書の `question` または `result` のJSON形式です。

# 開発の参考資料

## ローカルの Ollama を使う場合（低性能だが利用制限なし）：

- VsCode上でターミナルを開いて、以下を入力します。
```
ollama launch opencode --model=qwen3.5:0.8b
```

## クラウドの無料モデルを使う場合：(中性能、無料枠少ない)

- VsCode上でターミナルを開いて、 opencode と入力します。

- /models と入力し、Free 表示のあるモデルを選択します。（例: DeepSeek V4 Flash Free）

## Google AI Studioを使う場合:(高性能、無料枠多い)

- [Google AI Studio](https://aistudio.google.com/api-keys)を開きます。

- APIキーを作成、を押下し、キー名を適当に命名し、プロジェクトを新規作成します。

- APIキーが表示されるので、クリップボードにコピーしておきます。

- [プロジェクト一覧](https://aistudio.google.com/projects)を開き、新規作成したプロジェクトが無料枠となっていることを確認します。

- VsCode上でターミナルを開いて、 opencode と入力します。

- /connect と入力、プロバイダ一覧が表示されるので、Googleを選択、APIキーに先ほどのAPIキーを貼り付けます。

# AIを用いたコード修正

- opencodeに修正を依頼してみてください。（例：猫語で回答するボタンを追加して ）

- フロントエンド担当者は、html/JavaScriptを追加／修正して画面を構築してください。

- バックエンド担当者は、backend/app.py上にURLとAPIを作成してください。

# 参考リンク

- [Flask](https://flask.palletsprojects.com/en/stable/)

  - Python で書かれた Webアプリケーションサーバ

- [Vue.js](https://vuejs.org/)

  - JavaScript製製のWebフロントエンド フレームワーク

- [Vue.js Tutorial](https://ja.vuejs.org/tutorial/)

  - Vue.jsの入門用チュートリアル
  
- [OpenAI API](https://github.com/openai/openai-python)

  - Pythonから、OpenAI APIを呼び出すライブラリ

