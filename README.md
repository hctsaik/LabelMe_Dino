# video_annotator

video_annotator is a PyQt desktop tool for video/image annotation workflows.
It keeps LabelMe and X-AnyLabeling JSON compatibility, uses DINOv2 features plus
optical-flow propagation to extend anchor boxes across frames, and exports
frame images, annotations, `manifest.json`, and `export_package.zip`.

## Run

Copy `config.example.yaml` to `config.yaml`, then edit the local LabelMe or
X-AnyLabeling executable paths if you use those integrations.

```bat
run_gui.bat
```

Or run it through the external runtime:

```bat
.venv\Scripts\python.exe main.py
```

The Electron platform starts this tool as an external desktop-window module so
the PyQt image workflow stays native while the platform tracks readiness,
runtime diagnostics, logs, and process lifecycle.
