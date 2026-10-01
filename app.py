"""ASGI entry point for the browser-local live detector."""

from mobile_live import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8501)
