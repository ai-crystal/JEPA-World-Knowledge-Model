@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYEXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
set "BOOK=basics"

echo ============================================
echo   Same Map of the World - %BOOK%
echo ============================================
echo.

echo [1/5] figures ...
"%PYEXE%" books\build\render_figures.py --book %BOOK%
if errorlevel 1 goto fail

echo.
echo [2/5] navigation ...
"%PYEXE%" books\build\gen_nav.py
if errorlevel 1 goto fail

echo.
echo [3/5] compile PDF ...
"%PYEXE%" books\build\build.py --book %BOOK%
if errorlevel 1 goto fail

echo.
echo [4/5] check markdown ...
"%PYEXE%" books\build\checks\check_markdown.py --book %BOOK%

echo.
echo [5/5] check PDF text layer ...
"%PYEXE%" books\build\checks\verify_pdf_text.py output\v1-basics.pdf

echo.
echo Done: output\v1-basics.pdf
if /i not "%~1"=="silent" pause
exit /b 0

:fail
echo.
echo Build failed for %BOOK%.
pause
exit /b 1
