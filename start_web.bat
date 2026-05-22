@echo off
echo ========================================
echo   FuzzGAN Web Frontend
echo   基于对抗网络的协议模糊测试系统
echo ========================================
echo.

cd /d %~dp0

echo Installing dependencies...
pip install flask flask-cors -q

echo.
echo Starting web server...
echo Please open http://localhost:5000 in your browser
echo.

python web\app.py

pause
