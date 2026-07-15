from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional, List
import httpx, time, json, os, re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
from rag_system import kb, build_rag_prompt, KNOWLEDGE_BASE_DIR

app = FastAPI(title="PFE Anomaly API")

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_KEY = os.getenv("API_SECRET_KEY", "pfe-anomaly-2025")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/tmp/results"))
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def strip_think(raw):
    if isinstance(raw, str):
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    return raw

@app.on_event("startup")
async def startup():
    kb_dir = Path(os.getenv("KNOWLEDGE_BASE_DIR", str(KNOWLEDGE_BASE_DIR)))
    if kb_dir.exists(): 
        kb.build(kb_dir)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_ready": kb.is_ready,
        "chunks": len(kb.chunks),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/models")
def list_models():
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return {"available_models": [m["name"] for m in r.json().get("models", [])]}
    except Exception as e:
        return {"error": str(e), "available_models": []}

class AnomalyRequest(BaseModel):
    record: dict
    issues: list
    model: str = "mistral:7b"
    anomaly_type: Optional[str] = None
    topic: str = "retail.orders"

@app.post("/explain")
async def explain_anomaly(req: AnomalyRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    
    system_prompt, user_prompt = build_rag_prompt(
        req.record, req.issues, req.anomaly_type, req.topic)
    
    start = time.time()
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{OLLAMA_URL}/api/generate",
            json={
                "model": req.model,
                "system": system_prompt,
                "prompt": user_prompt,
                "stream": False,
                "format": "json"
            })
    data = r.json()
    latency_ms = round((time.time() - start) * 1000, 2)
    
    raw = strip_think(data.get("response", "{}"))
    try:
        explanation = json.loads(raw)
    except:
        explanation = {"raw": raw, "parse_error": True}
    
    result = {
        "model": req.model,
        "explanation": explanation,
        "latency_ms": latency_ms,
        "rag_chunks": len(kb.chunks),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    out = RESULTS_DIR / "raw" / f"{req.model.replace(':', '-')}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**result, "record": req.record, "issues": req.issues}, indent=2))
    
    return result

@app.post("/benchmark")
async def benchmark(req: AnomalyRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    
    models = os.getenv("MODELS", "mistral:7b").split(",")
    results = []
    for model in models:
        req.model = model.strip()
        try:
            r = await explain_anomaly(req, x_api_key=x_api_key)
            results.append(r)
        except Exception as e:
            results.append({"model": model, "error": str(e)})
    return {"results": results}

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "mistral:7b"
    system: str = ""
    messages: List[ChatMessage] = []

@app.post("/chat")
async def chat(req: ChatRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat",
            json={
                "model": req.model,
                "system": req.system,
                "messages": [{"role": m.role, "content": m.content} for m in req.messages],
                "stream": False
            })
    data = r.json()
    content = strip_think(data.get("message", {}).get("content", "No response"))
    return {"model": req.model, "message": {"role": "assistant", "content": content}}

class DebateRequest(BaseModel):
    record: dict
    issues: list
    anomaly_type: str
    pipeline_id: str = "retail"
    model: str = "mistral:7b"

@app.post("/debate")
async def run_debate(req: DebateRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    
    SYSTEMS = {
        "BusinessSpecialist": """You are a Carrefour Morocco business operations specialist.
State the EXACT business impact of this anomaly on this specific store in ONE sentence.
Respond ONLY in JSON: {"position":"one specific sentence","evidence":["reason1","reason2"],"confidence":0.0,"severity":"low|medium|high|critical","recommended_action":"one concrete action"}""",
        "DataQuality": """You are a data pipeline quality engineer.
State exactly which quality dimension is violated and which fields are affected.
Respond ONLY in JSON: {"position":"one specific sentence","evidence":["finding1","finding2"],"confidence":0.0,"severity":"low|medium|high|critical","recommended_action":"one concrete fix"}""",
        "AnomalyDetection": """You are a streaming pipeline anomaly detection specialist.
Assess true vs false positive and identify the pipeline component that caused this.
Respond ONLY in JSON: {"position":"one specific sentence","evidence":["evidence1","evidence2"],"confidence":0.0,"severity":"low|medium|high|critical","recommended_action":"one investigation step"}""",
        "Advisor": """You synthesize 3 specialist arguments into ONE final verdict.
agreed_fix must be ONE sentence not a list. Escalate if any agent confidence < 0.65.
Respond ONLY in JSON: {"agreed_cause":"one sentence","agreed_severity":"low|medium|high|critical","agreed_fix":"one sentence","confidence":0.0,"escalate_to_human":false,"escalation_reason":null,"manager_message":"max 2 plain sentences","engineer_message":"technical steps","debate_summary":"one sentence"}""",
    }
    
    start = time.time()
    record_str = json.dumps({k: v for k, v in req.record.items() if k not in ["lines", "_anomaly"]}, indent=2)
    arguments = {}
    
    for agent in ["BusinessSpecialist", "DataQuality", "AnomalyDetection"]:
        prompt = (f"Anomaly: {req.anomaly_type}\n"
                  f"Store: {req.record.get('store_name','?')} ({req.record.get('store_type','?')})\n"
                  f"Issues: {req.issues}\nRecord: {record_str}")
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{OLLAMA_URL}/api/generate",
                    json={"model": req.model, "system": SYSTEMS[agent], "prompt": prompt, "stream": False, "format": "json"})
            raw = strip_think(r.json().get("response", "{}"))
            arguments[agent] = json.loads(raw) if isinstance(raw, str) else raw
        except Exception as e:
            arguments[agent] = {"position": f"Error:{e}", "confidence": 0.3, "severity": "medium", "recommended_action": "Manual review"}
    
    args_text = "\n\n".join(f"=== {n} ===\n{json.dumps(a, indent=2)}" for n, a in arguments.items())
    
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_URL}/api/generate",
                json={"model": req.model, "system": SYSTEMS["Advisor"], "prompt": f"Agents analyzed {req.anomaly_type}:\n{args_text}\nSynthesize.", "stream": False, "format": "json"})
        raw = strip_think(r.json().get("response", "{}"))
        verdict = json.loads(raw) if isinstance(raw, str) else raw
    except Exception as e:
        verdict = {"agreed_cause": str(e), "agreed_severity": "medium", "agreed_fix": "Manual review", "confidence": 0.3,
                   "escalate_to_human": True, "escalation_reason": str(e), "manager_message": "Issue detected.", "engineer_message": str(e), "debate_summary": "Error"}
    
    conf = float(verdict.get("confidence", 0.5))
    if conf < 0.60:
        verdict["escalate_to_human"] = True
        verdict["escalation_reason"] = f"Low confidence {conf:.2f}"
    if verdict.get("agreed_severity") == "critical":
        verdict["escalate_to_human"] = True
    
    result = {
        "model": req.model,
        "pipeline_id": req.pipeline_id,
        "anomaly_type": req.anomaly_type,
        "verdict": verdict,
        "agent_arguments": arguments,
        "latency_ms": round((time.time() - start) * 1000, 2),
        "timestamp": datetime.utcnow().isoformat()
    }
    
    out = RESULTS_DIR / "debates" / f"debate_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result

@app.get("/kb/search")
def search_kb(q: str, k: int = 3, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    return {"query": q, "chunks": kb.retrieve(q, k=k)}

@app.post("/kb/reload")
def reload_kb(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(401, "Invalid API key")
    kb.update()
    return {"status": "reloaded", "chunks": len(kb.chunks)}
