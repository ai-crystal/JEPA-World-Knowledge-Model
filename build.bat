@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"
set "PYEXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"

set "BOOK=%~1"
if "%BOOK%"=="" set "BOOK=all"
if /i "%BOOK%"=="basics"   goto one
if /i "%BOOK%"=="practice" goto one
if /i "%BOOK%"=="theory"   goto one

echo [all] basics
call "%~dp0build-basics.bat" silent
if errorlevel 1 goto fail
echo [all] practice
call "%~dp0build-practice.bat" silent
if errorlevel 1 goto fail
echo [all] theory
call "%~dp0build-theory.bat" silent
if errorlevel 1 goto fail
echo.
echo All three volumes built. PDFs are in the output folder.
pause
exit /b 0

:one
call "%~dp0build-%BOOK%.bat"
exit /b %errorlevel%

:fail
echo.
echo Build failed.
pause
exit /b 1
