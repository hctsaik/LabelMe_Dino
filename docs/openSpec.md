# DLB × Electron 整合規格書（openSpec）

> 版本：1.0 | 日期：2026-05-17 | 適用：DLB Python 端實作

---

## 1. 整合概覽

### 架構決定

- **選項 1**：保留 PyQt5 GUI，用 PyInstaller 打包成 `dlb.exe`
- Electron 扮演「啟動器 + 結果接收者」
- 溝通：CLI 參數（啟動時）+ NDJSON over stdout（執行期）+ stdin commands
- LabelMe / X-AnyLabeling：外部 exe，不打包，路徑存 `config.yaml`

### 工作分工

| 元件 | 負責方 |
|------|--------|
| DINOv2 推理、Optical Flow propagation | DLB Python |
| LabelMe / X-AnyLabeling 呼叫 | DLB Python（subprocess） |
| JSON 讀寫（LabelMe 格式） | DLB Python |
| UI、流程控制、進度顯示 | Electron |
| 影片播放（smooth playback） | Electron 原生 `<video>` |

---

## 2. DLB Python 端需要的程式修改

### 2.1 新增 `--headless` 與 `--probe` 模式

**`main.py` 修改**

```python
import argparse
import json
import sys
from pathlib import Path

def _parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui",       action="store_true", default=True)
    group.add_argument("--headless",  action="store_true")
    group.add_argument("--probe",     action="store_true")

    parser.add_argument("--video",        default="")
    parser.add_argument("--output-dir",   default="")
    parser.add_argument("--config",       default="")
    parser.add_argument("--session-id",   default="")
    parser.add_argument("--log-file",     default="")
    parser.add_argument("--log-level",    default="info")
    parser.add_argument("--no-cache",     action="store_true")
    return parser.parse_args()

def main():
    args = _parse_args()

    if args.probe:
        _emit_probe()
        return

    log_file = _setup_logging(args.log_file, args.log_level)

    if args.headless:
        from src.headless import HeadlessWorker
        worker = HeadlessWorker(args)
        worker.run()          # blocks，讀 stdin loop
    else:
        # 原有 PyQt5 GUI 路徑
        app = QApplication(sys.argv)
        window = MainWindow(video_path=args.video)
        window.showMaximized()
        sys.exit(app.exec_())

def _emit_probe():
    import torch, platform
    probe = {
        "event": "probe",
        "dlb_version": "1.0.0",
        "protocol_version": 2,
        "capabilities": ["propagation", "labelme_export", "xanylabeling_export"],
        "python_version": sys.version.split()[0],
        "cuda_available": torch.cuda.is_available(),
        "dino_model": "dinov2-small",
        "platform": platform.system().lower(),
        "pid": os.getpid(),
    }
    print(json.dumps(probe, ensure_ascii=False), flush=True)
```

### 2.2 新增 `src/headless.py`（Electron 溝通核心）

```python
# src/headless.py
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.video_core import VideoCore
from src.dino_engine import DinoEngine
from src.propagator import Propagator, LabeledBox
from src.label_bridge import LabelBridge
from src.class_manager import ClassManager

class HeadlessWorker:
    def __init__(self, args):
        self.args = args
        self._seq = 0
        self._state = "initializing"
        self.session_id = args.session_id or ""

    def run(self):
        self._emit("ready", {
            "dlb_version": "1.0.0",
            "protocol_version": 2,
            "config_path": str(Path(self.args.config or "config.yaml").resolve()),
            "output_dir": self.args.output_dir,
            "video_path": self.args.video,
            "pid": os.getpid(),
        })
        self._state = "idle"

        # stdin 命令迴圈（主執行緒）
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                self._dispatch(cmd)
            except json.JSONDecodeError as e:
                self._emit_error("INVALID_JSON", str(e), recoverable=True)

    def _dispatch(self, cmd: dict):
        name = cmd.get("cmd", "")
        req_id = cmd.get("req_id", "")
        if name == "status":
            self._emit("status_response", {"state": self._state, "req_id": req_id})
        elif name == "load_video":
            self._do_load_video(cmd, req_id)
        elif name == "propagate":
            threading.Thread(target=self._do_propagate, args=(cmd, req_id), daemon=True).start()
        elif name == "export":
            threading.Thread(target=self._do_export, args=(cmd, req_id), daemon=True).start()
        elif name == "launch_tool":
            self._do_launch_tool(cmd, req_id)
        elif name == "shutdown":
            self._emit("shutdown", {"reason": "user_requested", "exit_code": 0})
            sys.exit(0)

    def _emit(self, event: str, data: dict = None):
        self._seq += 1
        msg = {
            "event": event,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seq": self._seq,
            **(data or {}),
        }
        print(json.dumps(msg, ensure_ascii=False), flush=True)

    def _emit_error(self, code: str, message: str, recoverable=True, **extra):
        self._emit("error" if recoverable else "fatal", {
            "code": code,
            "message": message,
            "recoverable": recoverable,
            **extra,
        })
```

### 2.3 `dino_engine.py` 修改：支援本地模型路徑

```python
import os, sys
from pathlib import Path

def _get_model_dir() -> str:
    env_dir = os.environ.get("DLB_MODEL_DIR", "")
    if env_dir and Path(env_dir).exists():
        return env_dir
    exe_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) \
              else Path(__file__).parent.parent
    local = exe_dir / "models" / "dinov2-small"
    if local.exists():
        return str(local)
    return "facebook/dinov2-small"

MODEL_ID = _get_model_dir()

class DinoEngine:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        is_local = Path(MODEL_ID).exists()
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=is_local)
        self.model = AutoModel.from_pretrained(MODEL_ID, local_files_only=is_local).to(self.device)
        self.model.eval()
```

### 2.4 `main.py` 路徑修正（打包環境）

```python
# 在 main.py 頂部，import torch 之前
import os, sys
from pathlib import Path

if getattr(sys, "frozen", False):
    _exe_dir = Path(sys.executable).parent
    # 強制離線模式（本地模型存在時）
    if (_exe_dir / "models" / "dinov2-small").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# 修正 config 和 log 路徑
CONFIG_PATH = str(Path(sys.executable).parent / "config.yaml") \
    if getattr(sys, "frozen", False) else "config.yaml"
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", ".")) / "DLB" / "logs" \
    if getattr(sys, "frozen", False) else Path("logs")
```

---

## 3. stdout 事件規格（NDJSON）

每行一個 JSON，以 `\n` 結尾。stderr 只寫 Python traceback，不混入業務 JSON。

### 基底欄位

```typescript
{ event, session_id, timestamp, seq }
```

### 完整事件清單

| event | 觸發時機 |
|-------|---------|
| `probe` | `--probe` 執行時輸出後退出 |
| `ready` | headless 初始化完成 |
| `video_loaded` | 影片 metadata 讀取完成 |
| `feature_extraction_start` | 開始計算 DINO features |
| `feature_extraction_progress` | 每 5% 或每 10 幀發一次 |
| `feature_extraction_done` | features 全部就緒 |
| `propagation_start` | 開始傳播 |
| `propagation_progress` | 每 10 幀發一次 |
| `propagation_done` | 傳播完成 |
| `export_start` | 開始匯出 |
| `export_done` | 匯出完成 |
| `tool_launched` | LabelMe/X-AnyLabeling 啟動成功 |
| `tool_exited` | 外部工具關閉 |
| `error` | 可恢復錯誤 |
| `fatal` | 不可恢復，DLB 即將退出 |
| `shutdown` | 正常退出前 |

### 關鍵事件 JSON Schema

```json
// video_loaded
{
  "event": "video_loaded",
  "session_id": "uuid",
  "timestamp": "2026-05-17T...",
  "seq": 3,
  "total_frames": 1800,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "duration_sec": 60.0,
  "cache_hit": false
}

// feature_extraction_progress
{
  "event": "feature_extraction_progress",
  "seq": 5,
  "processed_frames": 240,
  "total_frames": 1800,
  "percent": 13.3,
  "eta_sec": 34.8
}

// propagation_done
{
  "event": "propagation_done",
  "seq": 45,
  "total_frames_processed": 1800,
  "total_labels_propagated": 14823,
  "flagged_frames": 12,
  "elapsed_sec": 97.3,
  "output_annotation_path": "C:/output/annotations/"
}

// export_done
{
  "event": "export_done",
  "seq": 47,
  "format": "labelme_compatible",
  "output_dir": "C:/output/",
  "exported_frames": 1800,
  "elapsed_sec": 3.2
}

// error
{
  "event": "error",
  "seq": 7,
  "code": "FEATURE_EXTRACT_PARTIAL_FAIL",
  "severity": "warning",
  "message": "3 frames failed feature extraction, skipped",
  "recoverable": true
}
```

### 錯誤碼表

| 錯誤碼 | 說明 |
|--------|------|
| `VIDEO_OPEN_FAILED` | 影片無法開啟 |
| `VIDEO_CODEC_UNSUPPORTED` | 不支援的編解碼器 |
| `DINO_MODEL_LOAD_FAILED` | 模型載入失敗 |
| `DINO_CUDA_OOM` | GPU 記憶體不足 |
| `PROPAGATION_SEED_MISSING` | 找不到 anchor annotation |
| `EXPORT_WRITE_FAILED` | 輸出目錄寫入失敗 |
| `TOOL_NOT_FOUND` | 外部工具 exe 不存在 |
| `CONFIG_INVALID` | config.yaml 格式錯誤 |
| `INVALID_JSON` | stdin 命令格式錯誤 |

---

## 4. CLI 參數規格

```
dlb.exe [MODE] [OPTIONS]

模式（互斥）：
  --gui          啟動 PyQt5 GUI（預設）
  --headless     無 GUI，stdout JSON 模式（Electron 用）
  --probe        輸出能力 JSON 後退出

核心參數：
  --video <path>         影片路徑
  --output-dir <path>    輸出根目錄
  --config <path>        config.yaml 路徑（預設：exe 旁邊）
  --session-id <uuid>    Electron 指定的 session ID

行為參數：
  --log-file <path>      log 寫入路徑（不影響 stdout JSON）
  --log-level <level>    debug/info/warning/error
  --no-cache             禁用 feature cache

工具路徑覆蓋：
  --labelme-path <path>
  --xanylabeling-path <path>
```

**優先順序**：CLI 參數 > 環境變數（`DLB_*`） > `config.yaml` > 程式預設值

---

## 5. PyInstaller 打包規格

### 5.1 前置作業：預下載 DINOv2 模型

```python
# scripts/download_model.py
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="facebook/dinov2-small",
    local_dir="./models/dinov2-small",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
)
```

### 5.2 Runtime Hook（`scripts/runtime_hook.py`）

```python
import os, sys
from pathlib import Path

if getattr(sys, "frozen", False):
    _exe_dir = Path(sys.executable).parent

    # 1. HuggingFace 離線模式
    if (_exe_dir / "models" / "dinov2-small").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("DLB_MODEL_DIR", str(_exe_dir / "models" / "dinov2-small"))

    # 2. torch DLL 路徑
    _torch_lib = _exe_dir / "torch" / "lib"
    if _torch_lib.exists():
        os.add_dll_directory(str(_torch_lib))

    # 3. PyQt5 plugin 路徑
    _qt_plugins = _exe_dir / "PyQt5" / "Qt5" / "plugins"
    if _qt_plugins.exists():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(_qt_plugins / "platforms")
```

### 5.3 `dlb.spec`（關鍵設定）

```python
a = Analysis(
    ["main.py"],
    datas=[
        ("models", "models"),
        ("config.yaml", "."),
        (f"{venv_sp}/transformers", "transformers"),
        (f"{venv_sp}/PyQt5/Qt5/plugins", "PyQt5/Qt5/plugins"),
    ],
    hiddenimports=[
        "torch", "torch.nn", "torch._C", "torch.backends.cpu",
        "transformers.models.dinov2.modeling_dinov2",
        "transformers.models.dinov2.configuration_dinov2",
        "transformers.image_processing_utils",
        "safetensors", "safetensors.torch",
        "PyQt5", "PyQt5.QtWidgets", "PyQt5.QtCore", "PyQt5.sip",
        "watchdog", "watchdog.observers", "watchdog.observers.winapi",
        "cv2", "PIL", "PIL.Image", "yaml",
    ],
    excludes=["sympy", "torch.fx", "torch._dynamo", "matplotlib", "tkinter"],
    runtime_hooks=["scripts/runtime_hook.py"],
)

exe = EXE(..., console=False, upx=True,
    upx_exclude=["torch_cpu.dll", "c10.dll", "Qt5Core.dll", "Qt5Widgets.dll"])
```

### 5.4 預估大小與啟動時間

| 項目 | 預估 |
|------|------|
| dist/dlb/ 目錄（不含模型） | 900 MB ~ 1.2 GB |
| + DINOv2-small 模型 | + 88 MB |
| **總計** | **~1.0 ~ 1.3 GB** |
| 冷啟動時間（SSD） | 8 ~ 20 秒 |
| DINO model load | 3 ~ 6 秒（在 PrepareWorker 非同步，不阻塞 GUI） |

---

## 6. 安裝目錄結構

```
dist/dlb/                          # PyInstaller 輸出
  dlb.exe
  models/
    dinov2-small/
      config.json
      model.safetensors            # ~88 MB
      preprocessor_config.json
  config.yaml                      # 預設設定
  _internal/                       # PyInstaller runtime
    torch/lib/*.dll
    PyQt5/Qt5/plugins/
    ...

%APPDATA%\DLB\                     # 使用者資料（runtime 寫入）
  config.yaml                      # 使用者設定（覆蓋 exe 旁的預設）
  logs/
    dlb_YYYYMMDD_HHMMSS.log
  sessions/
    {session-uuid}/
      session.json
      dlb.log
      output/
        annotations/               # LabelMe JSON
        .cache/                    # feature .npy
```

---

## 7. config.yaml 管理原則

| 欄位 | 管理方 | 說明 |
|------|--------|------|
| `labelme.exe_path` | Electron 可寫 | 使用者設定路徑 |
| `x_anylabeling.exe_path` | Electron 可寫 | 使用者設定路徑 |
| `propagation.*` | DLB 管理 | 用 CLI 參數覆蓋，不直接寫 |
| `classes.*` | DLB 管理 | DLB GUI 才修改 |
| `session.tmp_dir` | DLB 管理 | runtime 決定 |

Electron 讀寫 config.yaml 時，只操作工具路徑欄位，其餘交由 DLB 管理。

---

*本文件由 SA 系統分析師、Vision AI 工程師、架構設計師三方討論後彙整*
