@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================================
echo  LabelMe_Dino - Build and Deploy
echo ============================================================

:: ── 0. 設定路徑 ─────────────────────────────────────────────
set CSC=C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe
set SRC_ROOT=%~dp0
set DIST_DEV=%SRC_ROOT%dist\LabelMe_Dino_launcher
:: Master repo is c:\code\claude\nativeApp (sibling of nativeApp_Int)
set MASTER_ROOT=%SRC_ROOT%..\..\nativeApp\LabelMe_Dino
set MASTER_DIST=%MASTER_ROOT%\dist\LabelMe_Dino_launcher

:: 確認 csc.exe 存在
if not exist "%CSC%" (
    echo [ERROR] csc.exe not found: %CSC%
    echo Please install .NET Framework 4.x
    pause & exit /b 1
)

:: ── 1. 編譯 C# Launcher ─────────────────────────────────────
echo.
echo [1/3] Compiling LabelMeDinoLauncher.cs ...
"%CSC%" /nologo /optimize /out:"%DIST_DEV%\LabelMe_Dino.exe" ^
    /target:exe ^
    /platform:anycpu ^
    launcher\LabelMeDinoLauncher.cs

if errorlevel 1 (
    echo [ERROR] Compilation failed.
    pause & exit /b 1
)
echo      OK - LabelMe_Dino.exe built

:: ── 2. 同步 Python 原始碼到 dev dist ────────────────────────
echo.
echo [2/3] Syncing source to dev dist ...

:: Root files
copy /y main.py       "%DIST_DEV%\main.py"       >nul
copy /y config.yaml   "%DIST_DEV%\config.yaml"   >nul
copy /y requirements.txt "%DIST_DEV%\requirements.txt" >nul
copy /y README.md     "%DIST_DEV%\README.md"     >nul

:: src/ - mirror with xcopy
xcopy /e /i /y /q src "%DIST_DEV%\src" >nul
echo      OK - source synced to dev dist

:: ── 3. 同步 dev dist 到 master 目錄 ─────────────────────────
echo.
echo [3/3] Deploying to master LabelMe_Dino ...

if not exist "%MASTER_DIST%" (
    echo [WARN] Master dist not found, skipping: %MASTER_DIST%
    goto :done
)

:: Exe
copy /y "%DIST_DEV%\LabelMe_Dino.exe"   "%MASTER_DIST%\LabelMe_Dino.exe"   >nul
copy /y "%DIST_DEV%\main.py"            "%MASTER_DIST%\main.py"            >nul
copy /y "%DIST_DEV%\config.yaml"        "%MASTER_DIST%\config.yaml"        >nul
copy /y "%DIST_DEV%\requirements.txt"   "%MASTER_DIST%\requirements.txt"   >nul
copy /y "%DIST_DEV%\README.md"          "%MASTER_DIST%\README.md"          >nul

:: src/
xcopy /e /i /y /q "%DIST_DEV%\src" "%MASTER_DIST%\src" >nul

echo      OK - deployed to master

:: 同時更新 master LabelMe_Dino 根目錄的 src/（供直接 run_gui.bat 使用）
xcopy /e /i /y /q src "%MASTER_ROOT%\src" >nul
copy /y main.py      "%MASTER_ROOT%\main.py"      >nul
copy /y config.yaml  "%MASTER_ROOT%\config.yaml"  >nul
copy /y requirements.txt "%MASTER_ROOT%\requirements.txt" >nul
copy /y README.md    "%MASTER_ROOT%\README.md"    >nul
echo      OK - master root src updated

:done
echo.
echo ============================================================
echo  Build complete!
echo  Dev dist:    %DIST_DEV%
echo  Master dist: %MASTER_DIST%
echo ============================================================
echo.
pause
