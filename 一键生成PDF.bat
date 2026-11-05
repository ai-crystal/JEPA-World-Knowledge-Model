@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYEXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

echo ============================================
echo   Same Map of the World - build script
echo ============================================
echo.

echo [1/5] generate figures ...
"%PYEXE%" book\build\make_covers.py
if errorlevel 1 goto fail
"%PYEXE%" book\build\render_figures.py
if errorlevel 1 goto fail

echo.
echo [2/5] update sidebar and README ...
"%PYEXE%" book\build\gen_nav.py
if errorlevel 1 goto fail

echo.
echo [3/5] compile PDF ...
"%PYEXE%" book\build\build_book.py
if errorlevel 1 goto fail

echo.
echo [4/5] check markdown ...
"%PYEXE%" book\build\checks\check_markdown.py

echo.
echo [5/5] check PDF text layer ...
"%PYEXE%" book\build\checks\verify_pdf_text.py

echo.
echo Done. PDF is in the output folder.
pause
exit /b 0

:fail
echo.
echo Build failed. Read the error above.
pause
exit /b 1
