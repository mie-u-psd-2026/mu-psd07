# 生成AI活用サンプルアプリ

# 概要

このアプリは Python とVue.jsを用いて作られた簡易的な生成AI活用アプリです。

- フロントエンドに、Vue.js CDN版を用いています。

- バックエンドはPython・Flaskで、既存のOpenAI SDKを通じてローカルのOllama native APIに接続します。

# フォルダ構成

```text
frontend/
  index.html          # Vue.jsによる画面・入力・通信
backend/
  app.py              # Flask API、モデル設定、画面の配信
  decision.py         # 判断条件の整理、生成、意味の審査
  ollama_client.py    # 文脈長を指定したOllama通信
  planning_prompt.txt # 候補、既知・不明・未確認の条件を整理する指示
  prompt.txt          # 選択式を優先した質問の生成指示
  result_prompt.txt   # 根拠と不確実性を明記する結果の生成指示
  review_prompt.txt   # 意味の審査の共通指示
  question_review_prompt.txt # 質問専用の審査基準
  result_review_prompt.txt   # 結果専用の審査基準
  requirements.txt    # Pythonの依存ライブラリ
  test_app.py         # APIと画面配信の自動テスト
  test_decision.py    # 計画・生成・審査と再試行の自動テスト
  evaluate.py         # 実モデルで会話を最後まで評価するツール
  eval_scenarios.json # 開発例以外の相談と利用者の設定
```

相談用AIは `qwen3:8b` を使用します。モデルの詳細は [Ollama公式ページ](https://ollama.com/library/qwen3:8b) を参照してください。
変更する場合は `backend/ollama_client.py` の既定値、または環境変数 `OLLAMA_MODEL` で指定します。
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
ollama pull qwen3:8b
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

通常は実用優先モードです。良い質問例を参考に生成し、合わない質問はスキップして進めます。検証結果は [evaluation.md](evaluation.md) を参照してください。

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
- 画面の状態遷移テスト：`node backend/test_frontend.cjs`（Node.jsがある場合）
- `/send_api` には `{"consultation":"相談内容","history":[]}` をPOSTします。履歴は `{"question":"質問","answer":"回答"}` の配列です。応答は仕様書の `question` または `result` のJSON形式です。

各回答から判断条件を整理し、次に確認する条件を決め、関連する質問例を最大2件参考にして質問を生成します。通常はAIによる別の意味審査で会話を止めず、形式・完全に同じ質問の重複・スキップ文面・選択肢の重複・推薦候補名をプログラムで確認します。質問はできるだけ選択肢で回答する形式にします。数値・自由入力は選択式では判断できない場合だけ使用します。

質問数の最低ノルマはありません。「わからない」が4回連続した場合は、別の条件を問い詰めず暫定結論へ進みます。十分な情報があれば結論を出し、最大15回答または「ここで結論を出す」でも結果を求められます。途中終了でも形式検証を行い、生成指示では不明な条件を事実で補わないよう求めます。結果の確信度は100%に固定しません。

既定のqwen3:8bは通常モードで使います。速度を優先し、通常は推論モードを無効にしています。

通常は1回のモデル呼び出しの中で条件整理を先に出力し、その後に選択式の質問または結論を生成します。不正な応答の修正を含め最大2回です。`DECISION_STRICT_REVIEW=true`で、従来の意味審査も必須にする検証用モードを有効にできます。修正を含め1操作全体で180秒、画面では190秒の期限があります。遅い端末では`DECISION_TIMEOUT_SECONDS`を30〜600秒の範囲で設定でき、画面の期限もサーバーの設定に合わせて10秒長くなります。応答の形式が不正な場合や通信エラーでは入力を保持して再試行できます。履歴を収めるため文脈長は8192で、環境変数`OLLAMA_NUM_CTX`で調整できます。

実モデルの会話評価と手順は [evaluation.md](evaluation.md) を参照してください。固定応答の自動テストだけで意味の正しさを保証せず、未知の相談・不明回答・会話全体の根拠を確認します。

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


質問の参考例は `backend/question_examples.json` で管理します。これはモデルの追加学習ではなく、実行時に関連する例を渡す方式です。回答済み・スキップ済みの同じ文面は参考例からも除外します。
