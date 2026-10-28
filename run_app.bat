@echo off
title PolyRegress Studio - Streamlit App Launcher
echo ========================================================
echo       Launching PolyRegress Studio Streamlit App
echo ========================================================
echo.

IF EXIST "%USERPROFILE%\anaconda3\Scripts\streamlit.exe" (
    echo Using Anaconda Python environment...
    "%USERPROFILE%\anaconda3\Scripts\streamlit.exe" run app.py
) ELSE IF EXIST "%USERPROFILE%\anaconda3\python.exe" (
    echo Using Anaconda Python executable...
    "%USERPROFILE%\anaconda3\python.exe" -m streamlit run app.py
) ELSE (
    echo Using system Python / Streamlit...
    streamlit run app.py
)

pause
