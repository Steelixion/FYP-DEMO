import asyncio
import json
import os
import time
import traceback
from typing import List, Optional, TypedDict

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

from langgraph.graph import StateGraph, END

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Using the EXACT model ID from your original voice.py file
MODEL_ID = "gemini-2.5-flash-native-audio-preview-12-2025" 

SYSTEM_PROMPT = """
You are "Echo", a highly sophisticated AI Driving Co-Pilot. 
Your primary mission is to ensure the driver stays awake and alert during their journey.
You have a friendly but authoritative personality—think of yourself as a trusted friend who is also a safety expert.

CRITICAL INSTRUCTION:
When you receive a [SYSTEM NOTIFICATION], you MUST speak to the driver IMMEDIATELY. 
Do not wait for them to talk. Intervene right away.

- If 'PRE-DROWSY': Start a conversation. "Hey, I noticed you're looking a bit tired. How are you holding up?"
- If 'DROWSY': Be loud and urgent. "Hey! Wake up! Stay focused on the road!"
- If 'YAWNING': "That was a big yawn! Do you need a break?"

Keep the driver talking. Ask them questions about their destination, their day, or tell them a joke. 
Always stay in character as a helpful, vigilant co-pilot.
"""

# --- DATA MODELS ---

class DrowsinessPayload(BaseModel):
    ear: float
    mar: float
    yaw: float
    tilt: float
    blink_rate: float
    risk: int
    status: str
    is_yawning: bool
    timestamp: Optional[float] = None

class AgentState(TypedDict):
    risk_score: int
    status: str
    is_yawning: bool
    intervention_needed: bool
    intervention_type: str
    last_message: str

# --- GLOBAL STATE ---
active_sessions: dict[WebSocket, asyncio.Queue] = {}
last_intervention_time = 0
INTERVENTION_COOLDOWN = 15

# --- LANGGRAPH AGENTS ---

def observer_agent(state: AgentState):
    global last_intervention_time
    risk = state["risk_score"]
    current_time = time.time()
    
    is_drowsy = risk >= 3 or state["is_yawning"]
    can_intervene = (current_time - last_intervention_time) > INTERVENTION_COOLDOWN
    
    intervention = is_drowsy and can_intervene
    
    itype = "none"
    if risk >= 6: itype = "urgent"
    elif risk >= 3: itype = "gentle"
    elif state["is_yawning"]: itype = "yawning"
        
    if intervention:
        last_intervention_time = current_time
    
    return {"intervention_needed": intervention, "intervention_type": itype}

def strategist_agent(state: AgentState):
    if not state["intervention_needed"]:
        return {"last_message": ""}
    
    itype = state["intervention_type"]
    if itype == "urgent":
        msg = "DRIVER IS DROWSY! WAKE THEM UP NOW!"
    elif itype == "gentle":
        msg = "DRIVER IS SHOWING PRE-DROWSY SIGNS. Start a conversation."
    elif itype == "yawning":
        msg = "DRIVER IS YAWNING. Suggest a break."
    else:
        msg = ""
        
    return {"last_message": msg}

async def communicator_agent(state: AgentState):
    msg = state.get("last_message", "")
    if msg:
        for queue in active_sessions.values():
            await queue.put(msg)
    return state

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("observer", observer_agent)
workflow.add_node("strategist", strategist_agent)
workflow.add_node("communicator", communicator_agent)
workflow.set_entry_point("observer")
workflow.add_edge("observer", "strategist")
workflow.add_edge("strategist", "communicator")
workflow.add_edge("communicator", END)
app_graph = workflow.compile()

# --- FASTAPI ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_ui():
    return FileResponse("index.html")

@app.post("/update")
async def update(payload: DrowsinessPayload):
    initial_state = {
        "risk_score": payload.risk,
        "status": payload.status,
        "is_yawning": payload.is_yawning,
        "intervention_needed": False,
        "intervention_type": "none",
        "last_message": ""
    }
    asyncio.create_task(app_graph.ainvoke(initial_state))
    return {"success": True}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Web client connected")
    
    trigger_queue: asyncio.Queue = asyncio.Queue()
    active_sessions[websocket] = trigger_queue

    # Reverting to the v1alpha version which is required for Live sessions
    client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1alpha"})
    
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(parts=[types.Part(text=SYSTEM_PROMPT)]),
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=False)
        ),
    )

    try:
        # Use the specific model name format that the API expects
        async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
            print("[GEMINI] Connected to Live API")
            
            # Start the conversation
            await trigger_queue.put("Hello! I am Echo, your AI co-pilot. I'm monitoring the road with you to keep you safe. How's the drive going so far?")

            async def browser_to_gemini():
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        )
                except Exception as e:
                    print(f"[WS] Browser loop closed: {e}")

            async def gemini_to_browser():
                try:
                    async for response in session.receive():
                        if response.server_content:
                            sc = response.server_content
                            if sc.model_turn:
                                for part in sc.model_turn.parts:
                                    if part.inline_data:
                                        await websocket.send_bytes(part.inline_data.data)
                            if sc.turn_complete:
                                await websocket.send_json({"type": "turn_complete"})
                except Exception as e:
                    print(f"[WS] Gemini loop closed: {e}")

            async def handle_triggers():
                try:
                    while True:
                        msg = await trigger_queue.get()
                        await session.send_client_content(
                            turns=[types.Content(
                                role="user",
                                parts=[types.Part(text=f"[SYSTEM NOTIFICATION]: {msg}")]
                            )],
                            turn_complete=True
                        )
                        trigger_queue.task_done()
                except Exception as e:
                    print(f"[WS] Trigger loop closed: {e}")

            await asyncio.gather(browser_to_gemini(), gemini_to_browser(), handle_triggers())

    except Exception as e:
        print(f"[SESSION ERROR] {e}")
        # DEBUG: Print available models to help troubleshoot 1008 error
        try:
            print("Checking available models...")
            for m in client.models.list():
                if "2.0-flash" in m.name:
                    print(f" - Found: {m.name}")
        except: pass
    finally:
        active_sessions.pop(websocket, None)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
