# FingerCursorController

人差し指を軽く曲げ、その側面（親指側）を親指でタップ／スライドする操作を、Webカメラ1台のハンドトラッキングで認識します。
認識結果（左/右/上/下スワイプ、タップ、長押し）はコンソールと画面に表示されるだけで、PC操作は行いません。

- ハンドトラッキング: MediaPipe Hand Landmarker
- 表示: OpenCV ウィンドウ（カメラ映像 + 骨格・接触距離・移動量などのオーバーレイ）
- ログ: 毎フレームの生ランドマーク・特徴量・状態遷移を JSONL で保存。ログだけから別パラメータで再認識できます

## セットアップ（Windows）

Python 3.11 以上（64bit）を用意し、PowerShell で:

```bash
cd FingerCursorController
```

```bash
py -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

初回起動時に MediaPipe のモデル（`models/hand_landmarker.task`）を自動でダウンロードします。

## 起動

```bash
python -m fingercursor
```

主なオプション:

| オプション | 内容 |
|---|---|
| `--guided` | ガイド付きテスト。画面の指示（DO: SWIPE LEFT など）どおりに操作すると、正解・誤認識・未検出を記録 |
| `--record` | カメラ映像を `video.mp4` としてセッションフォルダに保存 |
| `--input FILE` | カメラの代わりに動画ファイルを処理（`--record` で撮った映像の再処理など） |
| `--set KEY=VALUE` | 設定の一時上書き（例: `--set contact.touch_on=0.35`）。複数指定可 |
| `--verbose` | TOUCH_START などのデバッグイベントもコンソールに表示 |

カメラが開かない場合は `config.toml` の `camera.index` や `camera.backend = "msmf"` を試してください。

### キー操作（ウィンドウを選択した状態で）

| キー | 内容 |
|---|---|
| `q` / `Esc` | 終了 |
| `p` | 一時停止 / 再開 |
| `d` | デバッグパネル表示切替 |
| `←→↑↓` または `J L I K` | 「今そのスワイプをした」という正解ラベルを記録 |
| `t` / `h` | タップ / 長押しの正解ラベル |
| `Space` | 汎用マーカー（「ここで変な挙動があった」など） |

### 画面の見方

- **黄色の線**: 接触判定に使う人差し指の区間（太線が最も近い区間）
- **親指先の円**: 色が状態を表す（灰=IDLE, 橙=CANDIDATE, 緑=TOUCH, 紫=HOLDING, 水色=SWIPED）。緑の輪=`touch_on`、赤の輪=`touch_off` の距離
- **右上のバー**: 接触距離 `d/S`。緑線より下で接触、赤線より上で離脱
- **右のパッド**: 接触開始からの親指の移動量。水色=スワイプ閾値、緑=タップ許容、紫=長押し許容
- 左上: FPS・推論時間・状態・状態判定の理由（`swipe_ambiguous_axis` など）

## 認識ロジック

1. 手の大きさ `S` で距離を正規化（カメラとの距離に依存しない）
2. 親指先と人差し指の骨格線（付け根→第2関節→第1関節→指先）との最短距離 `d/S` で接触を判定（ヒステリシス + 連続フレーム確認）
3. 接触中、人差し指の付け根を基準にした親指先の位置変化（手全体の移動は相殺）を追跡
   - 移動量が `swipe_min_dist` を超え、方向が十分はっきりしていれば **スワイプ**（1回の接触で1回）
   - 動かずに `long_press_ms` 経過で **長押し開始**、離して **長押し終了**
   - 短時間で離せば **タップ**。どれにも該当しない接触は `TOUCH_REJECTED`（理由付き）としてログに残る

方向は既定で「画面上の見た目」（ミラー表示基準）です。手を傾けても方向を保ちたい場合は `direction.frame = "hand"`。

## ログ

起動ごとに `logs/YYYYMMDD_HHMMSS/` が作られます。

| ファイル | 内容 |
|---|---|
| `session.json` | 全設定・解像度・ライブラリバージョン・終了時の集計 |
| `frames.jsonl` | 毎フレーム: 時刻、処理時間、生ランドマーク21点（画像座標・world座標）、平滑化後の主要点、接触距離・最近区間・奥行き差・関節角度などの特徴量、状態機械の状態と判定理由 |
| `events.jsonl` | 認識イベント（ジェスチャー + TOUCH_START/END/REJECTED などのデバッグイベント。継続時間・移動量・最大速度・最小距離付き） |
| `labels.jsonl` | キーで付けた正解ラベル、`--guided` の試行結果 |
| `video.mp4` | `--record` 指定時のみ |

### うまく認識しないとき

1. `python -m fingercursor --guided` で一通り操作する（各ジェスチャー5回ずつ、約3分）
2. 終了後、作られたセッションフォルダ（`session.json`, `frames.jsonl`, `events.jsonl`, `labels.jsonl`）をまとめて渡してください。ガイドモードを使わない場合は、操作した直後に正解ラベルのキーを押しておくと分析しやすくなります
3. 自分で確認する場合:

```bash
python tools/summarize_log.py logs/20261029_120000
```

```bash
python tools/replay_log.py logs/20261029_120000 --set contact.touch_on=0.35 --events
```

```bash
python tools/replay_log.py logs/20261029_120000 --sweep contact.touch_on=0.25,0.3,0.35 --sweep gesture.swipe_min_dist=0.2,0.25,0.3
```

`summarize_log.py` は接触距離の分布・状態ごとの距離・接触の継続時間/移動量・失敗した試行を表示します。
`replay_log.py` は保存済みの生ランドマークからパラメータを変えて再認識し、正解ラベルとの一致率を比較します。

## テスト

```bash
python -m pytest
```
