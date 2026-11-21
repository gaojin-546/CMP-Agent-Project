@echo off
chcp 65001 >nul
setlocal
title CMP Agent

rem ============================================================
rem  CMP 抛光液配方开发 Agent —— 一键启动
rem    1) 自动切换到脚本所在目录（项目根）
rem    2) 启动 Streamlit，监听 0.0.0.0:8501（本机 + 局域网都可访问）
rem    3) 6 秒后自动用默认浏览器打开 http://127.0.0.1:8501
rem    4) 服务在前台运行，按 Ctrl+C 即可停止
rem ============================================================

rem ---- 1) 切到项目根目录（/d 兼容从其它盘符调用）----
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"

if not exist "src\app.py" (
    echo [错误] 当前目录下找不到 src\app.py
    echo        脚本所在目录：%~dp0
    pause
    exit /b 1
)

rem ---- 2) 选一个能用的 Python（优先 python，其次 py -3）----
set "PYCMD=python"
%PYCMD% -c "import streamlit" >nul 2>nul
if errorlevel 1 set "PYCMD=py -3"
%PYCMD% -c "import streamlit" >nul 2>nul
if errorlevel 1 goto :nodeps

rem ---- 3) 端口已被占用（重复双击）时只开浏览器，不启第二个实例 ----
netstat -ano | findstr /c:":8501 " | findstr /c:"LISTENING" >nul
if not errorlevel 1 (
    echo [提示] 8501 端口已有实例在运行，直接打开浏览器。
    start "" http://127.0.0.1:8501
    pause
    exit /b 0
)

echo CMP Agent 已成功启动，请不要关闭此黑框，按 Ctrl+C 退出。
echo.
echo   本机访问　：http://127.0.0.1:8501
echo   局域网访问：http://本机IP:8501
echo   数据目录　：%~dp0data\projects\
echo.

rem ---- 4) 后台延时 6 秒再开浏览器；主窗口保持前台跑服务（Ctrl+C 才能真正停掉它）----
start "" /b cmd /c "ping -n 7 127.0.0.1 >nul & start http://127.0.0.1:8501"

rem ---- 5) 前台启动 Streamlit ----
rem   --server.headless true 是必须的：否则首次运行会弹出 "Welcome to Streamlit!
rem   请输入邮箱" 的引导，在非交互环境下会直接退出（实测踩到）；它还会自己
rem   再开一个浏览器标签，与上面 start 的重复。
%PYCMD% -m streamlit run src\app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true

echo.
echo 服务已退出（返回码 %errorlevel%）。
pause
exit /b 0

:nodeps
echo [错误] 未检测到 streamlit，请先在本目录执行：
echo        pip install -r requirements.txt
pause
exit /b 1
