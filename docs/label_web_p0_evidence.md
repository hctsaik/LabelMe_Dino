# label_web P0 證據（2026-09-05）

未改 nativeApp 殼。實作在 `label_web/`。

## 測試

```text
PYTHONPATH=C:\code\claude\LabelMe_Dino
python -m pytest tests/test_label_web.py
→ 3 passed
```

涵蓋：sha256 任務清單、health `model_ready: false`、binary JPEG、sidecar 寫在 `images/<sha256>.json`、annotations 備份、`blind-quiz` 403 不寫檔、prepare_video 501。

## 真實 LV handoff 複本

來源（唯讀，未改原目錄）：

`C:\code\claude\LV\visuallatent\output\lv_labeling_handoff\gap_filler_20260621_052046_eda38d`

複本：`tmp/label_web_evidence/`

| 檢查 | 結果 |
|---|---|
| GET /health | `ok: true`, `model_ready: false`, `n: 1` |
| GET /api/task | sha256 `22b1d19abe62b4abfad0053cb432084aa8f2f2abfba5b8e8c72d6faad9d1bc3f` |
| GET /api/image/0 | `image/jpeg` 68509 bytes |
| POST /api/annotation/0 | 寫入 `images/<sha256>.json` 與 `annotations/<sha256>.json` |

拖框期間無 Python round-trip：前端 `mousemove` 只 `redraw()`，`POST` 只在按儲存。
