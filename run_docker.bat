@echo off

docker build -t msf_py3_image .

docker run --rm -it ^
  --cpus="4" ^
  --memory="8g" ^
  --memory-swap="8g" ^
  -v "%cd%":/app ^
  -w /app ^
  msf_py3_image main.py

pause