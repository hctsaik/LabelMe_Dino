@echo off
cd /d "%~dp0"
echo ============================================
echo  DINO-Labeler Bridge  ^|  環境安裝
echo ============================================

where python >nul 2>&1
if errorlevel 1 (
    echo [錯誤] 找不到 python，請先安裝 Python 3.11
    pause
    exit /b 1
)

echo [1/3] 建立虛擬環境 .venv ...
python -m venv .venv
if errorlevel 1 ( echo [錯誤] venv 建立失敗 & pause & exit /b 1 )

echo [2/3] 安裝套件 (torch + PyQt5 + labelme 等，約 2-4 GB)...
.venv\Scripts\pip install --upgrade pip
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 ( echo [錯誤] 套件安裝失敗 & pause & exit /b 1 )

echo [3/3] 完成！
echo.
echo 執行方式：雙擊 run_gui.bat
echo 首次啟動會自動下載 DINOv2 模型 (~80MB)，需要網路。
echo.
pause
