# label_web（殼內標註 P0）

NativeApp iframe 用的本機 HTTP。不 import PyQt。P1 抽幀、P2 DINOv2 傳播、GrabCut 分割已接上。`LABEL_WEB_SKIP_DINO=1` 可只抽幀不跑模型。

```powershell
cd C:\code\claude\LabelMe_Dino
$env:LABEL_WEB_TASK_DIR = "C:\path\to\lv_handoff"
$env:LABEL_WEB_DATA = "C:\path\to\writable"
python -m label_web --host 127.0.0.1 --port 8789
```

驗收：`python -m pytest tests/test_label_web.py tests/test_label_web_jobs.py`
