# DLB × Electron 整合指南

> 版本：1.0 | 日期：2026-05-17 | 受眾：Electron 開發團隊

---

## 1. 概覽

DLB（DINO-Labeler Bridge）是一個以 PyInstaller 打包的 Windows exe，提供：

- DINOv2 特徵提取 + Optical Flow 標註傳播
- LabelMe / X-AnyLabeling 外部工具呼叫
- YOLO / LabelMe 格式匯出

**整合方式**：Electron spawn `dlb.exe`，透過 stdout 接收 NDJSON 事件，透過 stdin 發送命令。dlb.exe 保留其 PyQt5 GUI（`--gui` 模式）或以無介面模式運作（`--headless` 模式）。

```
┌─────────────────────────────────┐
│           Electron              │
│  ┌──────────┐  ┌─────────────┐ │
│  │  Main    │  │  Renderer   │ │
│  │ Process  │  │  (React UI) │ │
│  └────┬─────┘  └──────┬──────┘ │
│       │  IPC           │        │
│  ┌────▼───────────────▼──────┐ │
│  │     DLBProcessManager     │ │
│  └────────────┬──────────────┘ │
└───────────────┼────────────────┘
                │ stdio
        ┌───────▼───────┐
        │   dlb.exe     │
        │  (headless)   │
        │               │
        │  DINOv2 model │
        │  OpenCV       │
        └───────────────┘
```

---

## 2. 快速開始

### 2.1 啟動流程

```javascript
// 標準啟動流程：probe → headless
const { spawn } = require('child_process');
const readline = require('readline');
const { v4: uuidv4 } = require('uuid');

async function startDLB(dlbExePath, videoPath, outputDir) {
  // Step 1: probe — 驗證 exe 能執行、取得能力
  const caps = await probeDLB(dlbExePath);
  console.log('DLB version:', caps.dlb_version);
  console.log('CUDA available:', caps.cuda_available);

  // Step 2: headless — 開始工作階段
  const sessionId = uuidv4();
  const proc = spawn(dlbExePath, [
    '--headless',
    '--video', videoPath,
    '--output-dir', outputDir,
    '--session-id', sessionId,
    '--log-level', 'info',
  ], {
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
  });

  // Step 3: 等待 ready 事件
  const ready = await waitForEvent(proc, 'ready');
  console.log('DLB ready, PID:', ready.pid);

  return { proc, sessionId };
}

async function probeDLB(dlbExePath) {
  return new Promise((resolve, reject) => {
    const proc = spawn(dlbExePath, ['--probe'], { stdio: 'pipe', windowsHide: true });
    let output = '';
    proc.stdout.on('data', d => output += d.toString());
    proc.on('close', code => {
      if (code !== 0) return reject(new Error(`probe exited ${code}`));
      try { resolve(JSON.parse(output.trim())); }
      catch { reject(new Error('probe output is not valid JSON')); }
    });
    setTimeout(() => { proc.kill(); reject(new Error('probe timeout')); }, 10000);
  });
}
```

### 2.2 NDJSON 解析器

```javascript
class DLBEventParser {
  constructor(proc) {
    this._handlers = new Map();
    this._rl = readline.createInterface({ input: proc.stdout });
    this._rl.on('line', line => {
      line = line.trim();
      if (!line) return;
      let evt;
      try { evt = JSON.parse(line); }
      catch { console.warn('[DLB] non-JSON stdout:', line); return; }
      const handlers = this._handlers.get(evt.event) || [];
      handlers.forEach(h => h(evt));
      const wildcards = this._handlers.get('*') || [];
      wildcards.forEach(h => h(evt));
    });
    // stderr 只輸出 Python traceback，記錄但不解析
    proc.stderr.on('data', d => console.error('[DLB stderr]', d.toString()));
  }

  on(eventName, handler) {
    if (!this._handlers.has(eventName)) this._handlers.set(eventName, []);
    this._handlers.get(eventName).push(handler);
    return this;
  }

  once(eventName) {
    return new Promise(resolve => {
      const handler = evt => {
        this._handlers.get(eventName)?.filter(h => h !== handler);
        resolve(evt);
      };
      this.on(eventName, handler);
    });
  }
}

// 使用範例
const parser = new DLBEventParser(proc);
parser.on('feature_extraction_progress', evt => {
  updateProgressBar(evt.percent, evt.eta_sec);
});
parser.on('error', evt => {
  if (!evt.recoverable) showFatalError(evt.message);
  else showWarning(evt.message);
});
```

### 2.3 發送命令到 stdin

```javascript
function sendCommand(proc, cmd, params = {}) {
  const reqId = uuidv4();
  const line = JSON.stringify({ cmd, req_id: reqId, ...params }) + '\n';
  proc.stdin.write(line);
  return reqId;
}

// 範例：執行傳播
sendCommand(proc, 'propagate', {
  start_frame: 0,
  end_frame: 1800,
  conf_threshold: 0.6,
});

// 範例：正常關閉
sendCommand(proc, 'shutdown');
```

---

## 3. 完整事件參考

所有事件都有以下基底欄位：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `event` | string | 事件名稱 |
| `session_id` | string | 由 Electron 傳入的 UUID |
| `timestamp` | string | ISO 8601 UTC |
| `seq` | number | 單調遞增序號（從 1 開始） |

### 3.1 `probe`

`dlb.exe --probe` 輸出一行 JSON 後退出（exit code 0）。

```json
{
  "event": "probe",
  "dlb_version": "1.0.0",
  "protocol_version": 2,
  "capabilities": ["propagation", "labelme_export", "xanylabeling_export"],
  "python_version": "3.11.9",
  "cuda_available": true,
  "dino_model": "dinov2-small",
  "platform": "windows",
  "pid": 12345
}
```

### 3.2 `ready`

headless 初始化完成，等待命令。

```json
{
  "event": "ready",
  "session_id": "abc123",
  "timestamp": "2026-05-17T10:00:00.000Z",
  "seq": 1,
  "dlb_version": "1.0.0",
  "protocol_version": 2,
  "config_path": "C:\\Users\\user\\AppData\\Roaming\\DLB\\config.yaml",
  "output_dir": "C:\\sessions\\abc123",
  "video_path": "C:\\videos\\demo.mp4",
  "pid": 12345
}
```

### 3.3 `video_loaded`

影片 metadata 讀取完成（在 `load_video` 命令後）。

```json
{
  "event": "video_loaded",
  "seq": 3,
  "total_frames": 1800,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "duration_sec": 60.0,
  "cache_hit": false
}
```

`cache_hit: true` 表示已有預先計算的 feature cache，`feature_extraction_*` 事件將被跳過。

### 3.4 特徵提取進度

```json
// feature_extraction_start
{ "event": "feature_extraction_start", "seq": 4, "total_frames": 1800 }

// feature_extraction_progress（每 5% 或每 10 幀一次）
{
  "event": "feature_extraction_progress",
  "seq": 5,
  "processed_frames": 240,
  "total_frames": 1800,
  "percent": 13.3,
  "eta_sec": 34.8
}

// feature_extraction_done
{
  "event": "feature_extraction_done",
  "seq": 24,
  "total_frames": 1800,
  "elapsed_sec": 42.1,
  "cache_path": "C:\\sessions\\abc123\\output\\.cache"
}
```

### 3.5 傳播進度

```json
// propagation_start
{ "event": "propagation_start", "seq": 25, "start_frame": 0, "end_frame": 1800 }

// propagation_progress（每 10 幀一次）
{
  "event": "propagation_progress",
  "seq": 30,
  "current_frame": 100,
  "total_frames": 1800,
  "percent": 5.6,
  "eta_sec": 82.4
}

// propagation_done
{
  "event": "propagation_done",
  "seq": 45,
  "total_frames_processed": 1800,
  "total_labels_propagated": 14823,
  "flagged_frames": 12,
  "elapsed_sec": 97.3,
  "output_annotation_path": "C:\\sessions\\abc123\\output\\annotations"
}
```

`flagged_frames`：AI 信心度低（< threshold）的幀數，建議提示使用者手動審查。

### 3.6 匯出

```json
// export_start
{ "event": "export_start", "seq": 46, "format": "labelme_compatible", "frame_count": 1800 }

// export_done
{
  "event": "export_done",
  "seq": 47,
  "format": "labelme_compatible",
  "output_dir": "C:\\sessions\\abc123\\output",
  "exported_frames": 1800,
  "elapsed_sec": 3.2
}
```

### 3.7 外部工具

```json
// tool_launched
{
  "event": "tool_launched",
  "seq": 6,
  "tool": "xanylabeling",
  "exe_path": "C:\\tools\\xanylabeling.exe",
  "frame_path": "C:\\videos\\frame_0042.jpg",
  "label_path": "C:\\sessions\\abc123\\output\\annotations\\frame_0042.json",
  "pid": 99876
}

// tool_exited
{
  "event": "tool_exited",
  "seq": 8,
  "tool": "xanylabeling",
  "exit_code": 0
}
```

### 3.8 錯誤

```json
// 可恢復錯誤（繼續運作）
{
  "event": "error",
  "seq": 7,
  "code": "FEATURE_EXTRACT_PARTIAL_FAIL",
  "severity": "warning",
  "message": "3 frames failed feature extraction, skipped",
  "recoverable": true
}

// 不可恢復錯誤（DLB 即將退出）
{
  "event": "fatal",
  "seq": 9,
  "code": "DINO_MODEL_LOAD_FAILED",
  "message": "Cannot load model from C:\\dlb\\models\\dinov2-small",
  "recoverable": false,
  "exit_code": 1
}
```

### 3.9 `shutdown`

```json
{
  "event": "shutdown",
  "seq": 99,
  "reason": "user_requested",
  "exit_code": 0
}
```

### 3.10 錯誤碼完整列表

| 錯誤碼 | 說明 | 可恢復 |
|--------|------|--------|
| `VIDEO_OPEN_FAILED` | 影片無法開啟（路徑錯誤/格式不支援） | ❌ |
| `VIDEO_CODEC_UNSUPPORTED` | 不支援的編解碼器 | ❌ |
| `DINO_MODEL_LOAD_FAILED` | 模型載入失敗（路徑不存在） | ❌ |
| `DINO_CUDA_OOM` | GPU 記憶體不足，自動降回 CPU | ✅ |
| `FEATURE_EXTRACT_PARTIAL_FAIL` | 部分幀提取失敗，已跳過 | ✅ |
| `PROPAGATION_SEED_MISSING` | 找不到 anchor annotation | ❌ |
| `PROPAGATION_PARTIAL_FAIL` | 部分幀傳播失敗，已跳過 | ✅ |
| `EXPORT_WRITE_FAILED` | 輸出目錄寫入失敗（磁碟空間/權限） | ❌ |
| `TOOL_NOT_FOUND` | 外部工具 exe 不存在 | ❌ |
| `CONFIG_INVALID` | config.yaml 格式錯誤 | ✅ |
| `INVALID_JSON` | stdin 命令格式錯誤 | ✅ |

---

## 4. stdin 命令參考

每行一個 JSON，以 `\n` 結尾。

### `status`

查詢目前狀態，DLB 回應 `status_response` 事件。

```json
{ "cmd": "status", "req_id": "uuid" }
```

回應：

```json
{ "event": "status_response", "state": "idle", "req_id": "uuid" }
```

`state` 可能值：`initializing` / `idle` / `loading_video` / `extracting` / `propagating` / `exporting` / `waiting_tool`

### `load_video`

載入影片並讀取 metadata（不啟動特徵提取）。

```json
{
  "cmd": "load_video",
  "req_id": "uuid",
  "video_path": "C:\\videos\\demo.mp4",
  "output_dir": "C:\\sessions\\abc123"
}
```

觸發事件序列：`video_loaded`（或 `error`）

### `propagate`

執行特徵提取（如需要）+ 傳播。非同步，事件驅動。

```json
{
  "cmd": "propagate",
  "req_id": "uuid",
  "start_frame": 0,
  "end_frame": 1800,
  "conf_threshold": 0.6,
  "use_cache": true
}
```

觸發事件序列：`feature_extraction_start` → `feature_extraction_progress` → `feature_extraction_done` → `propagation_start` → `propagation_progress` → `propagation_done`（或 `error`）

### `export`

匯出標註。

```json
{
  "cmd": "export",
  "req_id": "uuid",
  "format": "labelme_compatible",
  "output_dir": "C:\\export\\project1"
}
```

`format` 可選值：`labelme_compatible`（目前支援）、`yolo`（規劃中）

觸發事件序列：`export_start` → `export_done`（或 `error`）

### `launch_tool`

啟動外部標注工具（LabelMe / X-AnyLabeling）。

```json
{
  "cmd": "launch_tool",
  "req_id": "uuid",
  "tool": "xanylabeling",
  "frame_path": "C:\\videos\\frame_0042.jpg",
  "label_path": "C:\\sessions\\abc123\\output\\annotations\\frame_0042.json"
}
```

`tool` 可選值：`labelme` / `xanylabeling`

觸發事件：`tool_launched` → （使用者操作）→ `tool_exited`

### `shutdown`

正常關閉 DLB。

```json
{ "cmd": "shutdown", "req_id": "uuid" }
```

觸發事件：`shutdown`（然後 process exit 0）

---

## 5. CLI 啟動參數

```
dlb.exe [MODE] [OPTIONS]

模式（三選一，互斥）：
  --gui          啟動 PyQt5 GUI（預設，使用者直接操作）
  --headless     無 GUI，stdin/stdout JSON 模式（Electron 整合用）
  --probe        輸出能力 JSON 後立即退出

核心參數（headless 模式）：
  --video <path>           影片路徑（可選，也可後續用 load_video 命令傳入）
  --output-dir <path>      輸出根目錄
  --session-id <uuid>      Electron 指定的 session ID（用於 log 和事件標記）
  --config <path>          config.yaml 路徑（預設：exe 旁邊的 config.yaml）

行為參數：
  --log-file <path>        log 寫入路徑（預設：%APPDATA%\DLB\logs\）
  --log-level <level>      debug / info / warning / error（預設：info）
  --no-cache               禁用 feature cache

工具路徑覆蓋（優先於 config.yaml）：
  --labelme-path <path>
  --xanylabeling-path <path>
```

**配置優先順序**：CLI 參數 > 環境變數（`DLB_*`）> `%APPDATA%\DLB\config.yaml` > exe 旁的 `config.yaml` > 程式預設值

---

## 6. 建議的 JavaScript 模組設計

### 6.1 `DLBProcessManager`（Main Process）

```javascript
// src/main/dlb-process-manager.js
const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const readline = require('readline');
const path = require('path');

class DLBProcessManager extends EventEmitter {
  constructor({ dlbExePath, dlbDir }) {
    super();
    this._exePath = dlbExePath;
    this._dlbDir = dlbDir;
    this._proc = null;
    this._parser = null;
    this._sessionId = null;
    this._state = 'stopped';
    this._seq = 0;
  }

  get state() { return this._state; }

  async probe() {
    const proc = spawn(this._exePath, ['--probe'], {
      stdio: 'pipe',
      windowsHide: true,
      cwd: this._dlbDir,
    });
    return new Promise((resolve, reject) => {
      let out = '';
      proc.stdout.on('data', d => out += d);
      proc.on('close', code => {
        if (code !== 0) return reject(new Error(`probe exit ${code}`));
        try { resolve(JSON.parse(out.trim())); }
        catch { reject(new Error('invalid probe JSON')); }
      });
      setTimeout(() => { proc.kill(); reject(new Error('probe timeout')); }, 10000);
    });
  }

  async start({ videoPath, outputDir, sessionId, logLevel = 'info' }) {
    if (this._proc) throw new Error('DLB already running');
    this._sessionId = sessionId;

    this._proc = spawn(this._exePath, [
      '--headless',
      '--video', videoPath || '',
      '--output-dir', outputDir,
      '--session-id', sessionId,
      '--log-level', logLevel,
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
      cwd: this._dlbDir,
    });

    this._proc.on('close', code => {
      this._state = 'stopped';
      this._proc = null;
      this.emit('stopped', { code });
    });

    this._setupParser();
    return this._waitEvent('ready', 30000);
  }

  send(cmd, params = {}) {
    if (!this._proc) throw new Error('DLB not running');
    const reqId = require('crypto').randomUUID();
    const payload = JSON.stringify({ cmd, req_id: reqId, ...params }) + '\n';
    this._proc.stdin.write(payload);
    return reqId;
  }

  stop() {
    if (!this._proc) return;
    this.send('shutdown');
    setTimeout(() => { this._proc?.kill(); }, 5000);
  }

  _setupParser() {
    const rl = readline.createInterface({ input: this._proc.stdout });
    rl.on('line', line => {
      line = line.trim();
      if (!line) return;
      let evt;
      try { evt = JSON.parse(line); }
      catch { return; }
      this._state = this._stateFromEvent(evt.event);
      this.emit('event', evt);
      this.emit(evt.event, evt);
    });
    this._proc.stderr.on('data', d => {
      this.emit('stderr', d.toString());
    });
  }

  _waitEvent(name, timeoutMs) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(`timeout waiting for ${name}`)), timeoutMs);
      this.once(name, evt => { clearTimeout(timer); resolve(evt); });
      this.once('fatal', evt => { clearTimeout(timer); reject(new Error(evt.message)); });
    });
  }

  _stateFromEvent(event) {
    const map = {
      ready: 'idle',
      video_loaded: 'idle',
      feature_extraction_start: 'extracting',
      feature_extraction_done: 'idle',
      propagation_start: 'propagating',
      propagation_done: 'idle',
      export_start: 'exporting',
      export_done: 'idle',
      tool_launched: 'waiting_tool',
      tool_exited: 'idle',
      shutdown: 'stopped',
    };
    return map[event] || this._state;
  }
}

module.exports = { DLBProcessManager };
```

### 6.2 IPC 通道命名慣例

Main Process ↔ Renderer 之間的 IPC channel 命名：

| 方向 | Channel | 用途 |
|------|---------|------|
| Renderer → Main | `dlb:start` | 啟動 headless session |
| Renderer → Main | `dlb:command` | 轉發 stdin 命令 |
| Renderer → Main | `dlb:stop` | 正常關閉 |
| Main → Renderer | `dlb:event` | 轉發所有 stdout 事件 |
| Main → Renderer | `dlb:state` | state 變更通知 |
| Main → Renderer | `dlb:error` | 錯誤通知 |

```javascript
// preload.js
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('dlb', {
  start: (opts) => ipcRenderer.invoke('dlb:start', opts),
  command: (cmd, params) => ipcRenderer.invoke('dlb:command', { cmd, params }),
  stop: () => ipcRenderer.invoke('dlb:stop'),
  onEvent: (handler) => ipcRenderer.on('dlb:event', (_, evt) => handler(evt)),
  offEvent: (handler) => ipcRenderer.off('dlb:event', handler),
});
```

```javascript
// main.js
const { ipcMain } = require('electron');

const manager = new DLBProcessManager({ dlbExePath, dlbDir });

manager.on('event', evt => {
  mainWindow.webContents.send('dlb:event', evt);
});

ipcMain.handle('dlb:start', (_, opts) => manager.start(opts));
ipcMain.handle('dlb:command', (_, { cmd, params }) => manager.send(cmd, params));
ipcMain.handle('dlb:stop', () => manager.stop());
```

### 6.3 React Renderer 使用範例

```jsx
// src/renderer/hooks/useDLB.js
import { useEffect, useRef, useState } from 'react';

export function useDLB() {
  const [state, setState] = useState('stopped');
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    const handler = (evt) => {
      switch (evt.event) {
        case 'feature_extraction_progress':
        case 'propagation_progress':
          setProgress({ percent: evt.percent, eta: evt.eta_sec });
          break;
        case 'propagation_done':
          setProgress(null);
          setState('idle');
          break;
        case 'fatal':
          setState('error');
          break;
      }
    };
    window.dlb.onEvent(handler);
    return () => window.dlb.offEvent(handler);
  }, []);

  const startPropagation = (startFrame, endFrame) => {
    window.dlb.command('propagate', {
      start_frame: startFrame,
      end_frame: endFrame,
      conf_threshold: 0.6,
    });
    setState('propagating');
  };

  return { state, progress, startPropagation };
}
```

---

## 7. config.yaml 結構

DLB exe 旁邊（或 `%APPDATA%\DLB\`）的 `config.yaml`：

```yaml
labelme:
  exe_path: ""            # Electron 設定精靈填寫
  debounce_ms: 500

x_anylabeling:
  exe_path: ""            # Electron 設定精靈填寫

propagation:
  conf_threshold: 0.6
  of_weight: 0.5
  dino_weight: 0.5

classes: []               # 標注類別，由 DLB GUI 管理
```

**Electron 只寫 `labelme.exe_path` 和 `x_anylabeling.exe_path`**，其餘欄位由 DLB 管理。

讀寫 config.yaml 的推薦做法：

```javascript
const yaml = require('js-yaml');
const fs = require('fs');

function setToolPath(configPath, tool, exePath) {
  let cfg = {};
  try { cfg = yaml.load(fs.readFileSync(configPath, 'utf8')) || {}; } catch {}
  if (!cfg[tool]) cfg[tool] = {};
  cfg[tool].exe_path = exePath;
  fs.writeFileSync(configPath, yaml.dump(cfg), 'utf8');
}

// 範例
setToolPath(configPath, 'x_anylabeling', 'C:\\tools\\xanylabeling.exe');
```

---

## 8. 安裝目錄結構

```
dist/dlb/                          # 發布套件（與 Electron installer 一起）
  dlb.exe
  config.yaml                      # 預設設定（初次安裝用）
  models/
    dinov2-small/
      config.json
      model.safetensors            # ~88 MB
      preprocessor_config.json
  _internal/                       # PyInstaller runtime（勿修改）
    torch/lib/*.dll
    PyQt5/Qt5/plugins/
    ...

%APPDATA%\DLB\                     # 使用者資料（runtime 產生）
  config.yaml                      # 使用者設定（覆蓋 dist 裡的預設）
  logs/
    dlb_YYYYMMDD_HHMMSS.log
  sessions/
    {session-uuid}/
      session.json
      dlb.log
      output/
        annotations/               # LabelMe JSON（每幀一個）
        .cache/                    # DINOv2 feature cache（.npy）
```

---

## 9. 初次設定精靈（Setup Wizard）建議

首次啟動時，Electron 應引導使用者完成：

1. **probe 驗證**：確認 `dlb.exe` 路徑正確且可執行
2. **CUDA 狀態提示**：`probe.cuda_available = false` 時，提示使用者 CPU 模式較慢
3. **LabelMe 路徑**：讓使用者選擇 `labelme.exe` 路徑，寫入 config.yaml
4. **X-AnyLabeling 路徑**：同上

```javascript
async function runSetupWizard(dlbExePath, configPath) {
  // 1. 驗證 dlb.exe
  const caps = await probeDLB(dlbExePath);

  // 2. CUDA 提示
  if (!caps.cuda_available) {
    await showWarning('GPU 加速不可用，處理速度較慢（CPU 模式）');
  }

  // 3. LabelMe 路徑
  const { filePaths } = await dialog.showOpenDialog({
    title: '選擇 LabelMe 執行檔',
    filters: [{ name: 'Executable', extensions: ['exe'] }],
  });
  if (filePaths.length) setToolPath(configPath, 'labelme', filePaths[0]);

  // 4. X-AnyLabeling 路徑（可跳過）
  const xanyResult = await showOptionalFileDialog('選擇 X-AnyLabeling 執行檔（可選）');
  if (xanyResult) setToolPath(configPath, 'x_anylabeling', xanyResult);
}
```

---

## 10. 錯誤處理與重啟策略

| 情況 | 建議處理 |
|------|---------|
| `fatal` 事件 | 顯示錯誤對話框，提供「查看 log」按鈕；不自動重啟 |
| `error` 事件（`recoverable: true`）| 顯示非阻塞通知（toast），繼續工作 |
| process crash（非正常退出） | 等待 `close` 事件，code ≠ 0 → 顯示崩潰對話框，提供 log 路徑 |
| `DINO_CUDA_OOM` | 提示「切換 CPU 模式」，用 `--device cpu` 重新啟動 |
| probe timeout | 提示 exe 路徑錯誤或被防毒軟體封鎖 |
| `TOOL_NOT_FOUND` | 重新開啟設定精靈，讓使用者更新路徑 |

```javascript
manager.on('stopped', ({ code }) => {
  if (code !== 0) {
    const logPath = path.join(
      process.env.APPDATA, 'DLB', 'logs'
    );
    showCrashDialog({ code, logPath });
  }
});

manager.on('fatal', (evt) => {
  showErrorDialog({
    title: 'DLB 發生不可恢復的錯誤',
    message: evt.message,
    code: evt.code,
  });
});
```

---

## 11. 效能預期

| 項目 | 預期值 | 備註 |
|------|--------|------|
| `dlb.exe` 冷啟動 | 8 ~ 20 秒 | SSD，PyInstaller 解壓 |
| probe 回應 | < 10 秒 | 含 exe 冷啟動 |
| DINOv2 模型載入 | 3 ~ 6 秒 | 在背景執行，不阻塞 ready 事件 |
| Feature 提取（CPU） | ~0.5 秒/幀 | 60 秒影片 → ~15 分鐘 |
| Feature 提取（GPU） | ~0.05 秒/幀 | 60 秒影片 → ~1.5 分鐘 |
| 傳播（含 OF） | ~0.05 秒/幀 | CPU |
| 匯出（1800 幀） | 3 ~ 5 秒 | I/O bound |
| dist/dlb/ 目錄大小 | ~1.0 ~ 1.3 GB | 含 DINOv2 模型 |

---

## 12. 常見問題

**Q: 如何知道 DLB 已準備好接受命令？**  
A: 等待 `ready` 事件（seq=1）。在此之前不要發送任何命令。

**Q: headless 模式下還會有視窗出現嗎？**  
A: 不會。`--headless` 不初始化 Qt，完全無視窗。但使用者如果需要操作 LabelMe 或 X-AnyLabeling，DLB 會呼叫那些外部 exe，那些 exe 有自己的視窗。

**Q: 如何傳入影片路徑？**  
A: 兩種方式：(1) 啟動時 `--video <path>`，(2) 啟動後發送 `load_video` 命令。建議用方式 2，讓使用者在 Electron UI 選擇影片後再傳入。

**Q: stderr 有 JSON 嗎？**  
A: 沒有。stderr 只有 Python traceback，用於除錯。所有業務事件都在 stdout NDJSON。

**Q: DLB 支援多個同時執行的 session 嗎？**  
A: 不支援。每個 dlb.exe process 處理一個 session。Electron 若需要多個 session，需 spawn 多個 process（各用不同 `--output-dir` 和 `--session-id`）。

**Q: 如何取得目前正在處理的狀態？**  
A: 發送 `status` 命令，DLB 回應 `status_response`。

**Q: cache 存在哪？**  
A: `{output_dir}/.cache/`，以影片路徑 hash 為檔名（`.npy` 格式）。再次處理同一影片時自動使用。

---

*如有問題，請聯絡 DLB Python 端維護者，或參考 `docs/openSpec.md` 取得更多實作細節。*
