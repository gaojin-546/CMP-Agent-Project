@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title CMP RAG Service (Python 3.11)

rem ============================================================
rem  CMP 知识库微服务（RAG）
rem    1) 启动前清理占用 8770 的残留进程（幽灵 python）
rem    2) 预热检查：Python 3.11 / 脚本文件
rem    3) 前台运行服务，任何情况下结束都 pause，不会闪退
rem ============================================================

cd /d "%~dp0"

set "PORT=8770"

echo ============================================================
echo   CMP RAG 知识库服务
echo   地址：http://127.0.0.1:%PORT%
echo   知识库：E:\private_rag_cmp\private_kb
echo   模型：  E:\private_rag_cmp\model\bge-m3
echo ============================================================
echo.

rem ---- 1) 端口占用检查：发现残留进程就强制结束 ----
set "KILLED="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do (
    echo [清理] 端口 %PORT% 被 PID %%p 占用，正在结束…
    taskkill /F /PID %%p >nul 2>nul
    if errorlevel 1 (
        echo [清理] taskkill 被拒绝，改用 PowerShell 结束 %%p …
        powershell -NoProfile -Command "Stop-Process -Id %%p -Force -ErrorAction SilentlyContinue" >nul 2>nul
    )
    set "KILLED=1"
)
if defined KILLED (
    echo [清理] 等待端口释放…
    ping -n 3 127.0.0.1 >nul
)

rem ---- 1b) 复查：仍被占用就明确告知并退出，绝不启第二个实例 ----
set "STILL="
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT%" ^| findstr "LISTENING"') do set "STILL=%%p"
if defined STILL (
    echo.
    echo [警告] 端口 %PORT% 仍被 PID !STILL! 占用，无法自动结束。
    echo        常见原因：那个进程属于别的用户，或权限更高。
    echo        请任选其一：
    echo         ① 以管理员身份重新运行本 bat
    echo         ② 手动执行 taskkill /F /PID !STILL!
    echo        然后重新双击本脚本。
    goto :end
)

rem ---- 2) 前置检查：Python 3.11 / 脚本文件 ----
py -3.11 -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo [错误] 找不到 Python 3.11（py -3.11）。
    echo        RAG 服务必须用 3.11 运行，因为 chromadb 在 3.15 上装不上。
    echo        本机已安装的 Python：
    py -0p
    goto :end
)

if not exist "src\rag_service.py" (
    echo [错误] 找不到 src\rag_service.py
    echo        请确认本 bat 与 src 目录在同一个项目根目录下。
    echo        当前目录：%CD%
    goto :end
)

rem ---- 3) 启动服务（前台运行，Ctrl+C 停止）----
echo [启动] 加载 BGE-M3（约 2.27 GB，首次 9-40 秒）…
echo [启动] 看到「RAG 服务就绪」后再去刷新网页。
echo.
py -3.11 -X utf8 src\rag_service.py --port %PORT%

echo.
echo 服务已退出（返回码 %errorlevel%）。
echo 如果上面有报错，请把它整段发给我；窗口不会自动关闭。

:end
echo.
pause
endlocal
