# Android / Termux

Use `mobile_live.py` for the phone workflow. It receives live camera frames from
the Android browser, runs YOLO inference in the Python process, saves MP4 clips
while a person is visible, and sends clips to Telegram after the person leaves.
`local_test.py` uses a desktop webcam and OpenCV display window, so it is not the
Android entry point.

## Install

PyTorch's standard Linux packages need glibc, while native Termux uses Android's
different C library. The recommended attempt is to run a Debian userspace inside
Termux with `proot-distro`.

In Termux, install Debian:

```sh
pkg update -y
pkg install proot-distro
proot-distro install debian
```

In Termux, grant storage access and copy the project folder (including
`yolov8n.pt`) from Downloads into Termux home:

```sh
termux-setup-storage
mkdir -p ~/live-detect
cp -r ~/storage/shared/Download/live-detect/. ~/live-detect/
```

Enter Debian with the project folder mounted at `/root/live-detect`:

```sh
proot-distro login debian --bind "$HOME/live-detect:/root/live-detect"
```

Then install its Python tools:

```sh
apt update
apt install -y python3 python3-venv python3-pip libgl1 libglib2.0-0 libgomp1
cd /root/live-detect
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model and its dependencies are large. Leave several gigabytes of free
storage, and expect CPU inference to be slower and use more battery than a
desktop GPU.

## Verify and launch

First verify that the device can import PyTorch and load the bundled model:

```sh
python -c "import torch; from ultralytics import YOLO; YOLO('/root/live-detect/yolov8n.pt'); print('PyTorch and model loaded:', torch.__version__)"
```

If installation or this check fails with an unsupported wheel, shared-library,
or architecture error, that Android/Python combination cannot run this model
through the current Ultralytics setup. Do not run `local_test.py` as a workaround;
it also requires a desktop-style camera and display.

Start the live mobile app from the project directory:

```sh
uvicorn mobile_live:app --host 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501` in the Android browser, press **Start camera**, and
allow camera access. Keep the terminal session running. Recordings are saved in
`recordings/` in the project directory.

## Optional Telegram alerts

The bot token was previously embedded in the source files. Revoke it and create
a replacement with BotFather before using Telegram alerts. In the Debian shell,
enter the replacement token without echoing it, then set the chat ID:

```sh
read -rsp "Telegram bot token: " TELEGRAM_BOT_TOKEN
printf '\n'
export TELEGRAM_BOT_TOKEN
export TELEGRAM_CHAT_ID="your-chat-id"
```

Keep that shell session open while Streamlit runs. Telegram alerts are optional;
the detector works without them.