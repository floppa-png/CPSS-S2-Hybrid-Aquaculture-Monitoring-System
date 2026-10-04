from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import json
from datetime import datetime
from typing import Optional

from agent.agent import agent_assessment, simulate_tomorrow

app = FastAPI(
    title="Aquaculture WQMS API",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Live sensor state — updated whenever ESP32 pushes a reading ─────────
latest_sensor: dict = {
    "connected":  False,
    "updated_at": None,
    "data":       None,
}


@app.websocket("/ws")
async def esp32_websocket(websocket: WebSocket):
    await websocket.accept()
    latest_sensor["connected"] = True
    print("[WS] ESP32 connected.")
    try:
        while True:
            text = await websocket.receive_text()
            payload = json.loads(text)
            latest_sensor["data"] = {
                "Temp":      float(payload["Temp"]),
                "pH":        float(payload["pH"]),
                "Turbidity": float(payload["Turbidity"]),
                "TDS":       float(payload["TDS"]),
            }
            latest_sensor["updated_at"] = datetime.utcnow().isoformat()
            print(f"[WS] Received: {latest_sensor['data']}")
    except WebSocketDisconnect:
        latest_sensor["connected"] = False
        print("[WS] ESP32 disconnected.")
    except Exception as e:
        latest_sensor["connected"] = False
        print(f"[WS] Error: {e}")


@app.get("/latest-sensor")
def get_latest_sensor():
    return latest_sensor


class WaterInput(BaseModel):
    Temp:      float
    pH:        float
    Turbidity: float
    TDS:       float
    fish_type: Optional[str] = "Tilapia"
    do_lag1:   Optional[float] = None
    temp_lag1: Optional[float] = None
    ph_lag1:   Optional[float] = None
    turbidity_lag1: Optional[float] = None
    tds_lag1:  Optional[float] = None


class ChatInput(BaseModel):
    message: str


@app.post("/predict")
def predict(data: WaterInput):
    inputs = {
        "Temp":      data.Temp,
        "pH":        data.pH,
        "Turbidity": data.Turbidity,
        "TDS":       data.TDS,
        "do_lag1":   data.do_lag1,
        "temp_lag1": data.temp_lag1,
        "ph_lag1":   data.ph_lag1,
        "turbidity_lag1": data.turbidity_lag1,
        "tds_lag1":  data.tds_lag1,
    }

    today = agent_assessment(inputs, data.fish_type)
    tomorrow_inputs = simulate_tomorrow(inputs)
    tomorrow = agent_assessment(tomorrow_inputs, data.fish_type)

    return {
        "today":    today,
        "tomorrow": tomorrow,
    }


@app.post("/chat")
def chat(chat: ChatInput):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":  "llama3.1:8b-instruct-q4_K_M",
                "prompt": chat.message,
                "stream": False,
                "options": {"temperature": 0.3, "num_ctx": 2048},
            },
            timeout=30,
        )
        return {"response": response.json()["response"].strip()}
    except Exception:
        return {"response": "Local AI assistant unavailable. Ensure Ollama is running."}


app.mount(
    "/",
    StaticFiles(directory="app/static", html=True),
    name="static",
)
