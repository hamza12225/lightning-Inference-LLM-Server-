#!/bin/bash
set -uo pipefail

STUDIO="/teamspace/studios/this_studio"
PROJECT="$STUDIO/pfe-llm-anomaly"
LOGS="$PROJECT/logs"
APP="$PROJECT/app"

mkdir -p "$LOGS"

echo "=============================================="
echo " PFE · Session Start — $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# --- 1. Ollama ---
echo "[1/5] Installing Ollama binary..."
curl -fsSL https://ollama.com/install.sh | sh > "$LOGS/ollama_install.log" 2>&1
echo " ✓ Ollama installed"

# --- 2. Python packages ---
echo "[2/5] Installing Python packages..."
pip install --quiet fastapi uvicorn httpx pydantic python-dotenv faiss-cpu sentence-transformers numpy
echo " ✓ packages ready"

# --- 3. Start Ollama ---
echo "[3/5] Starting Ollama..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 2
nohup ollama serve > "$LOGS/ollama.log" 2>&1 &
disown

OLLAMA_UP=false
for i in {1..25}; do
    if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
        OLLAMA_UP=true
        break
    fi
    sleep 2
done

if [ "$OLLAMA_UP" != true ]; then
    echo " ✗ Ollama failed to start. Check: cat $LOGS/ollama.log"
    exit 1
fi

MODEL_COUNT=$(ollama list 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
MODEL_NAMES=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' ')
echo " ✓ Ollama ready — $MODEL_COUNT models: $MODEL_NAMES"

# Must match the MODELS list in appi.py exactly, or the dashboard will offer
# models that don't exist in Ollama and every call to them will fail.
REQUIRED_MODELS=("mistral:7b" "llama3.1:8b" "phi4-mini:latest" "qwen3:8b")

for m in "${REQUIRED_MODELS[@]}"; do
    if ! echo "$MODEL_NAMES" | grep -qw "$m"; then
        echo " Pulling missing model: $m ..."
        ollama pull "$m" 2>&1 | tail -3
    fi
done

MODEL_COUNT=$(ollama list 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')
MODEL_NAMES=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ' ')
echo " ✓ All required models present — $MODEL_COUNT models: $MODEL_NAMES"

# --- 4. Start FastAPI + RAG ---
echo "[4/5] Starting FastAPI + RAG..."
pkill -f uvicorn 2>/dev/null || true
sleep 2

if [ ! -f "$APP/main.py" ]; then
    echo " ✗ $APP/main.py not found. Aborting."
    exit 1
fi

cd "$APP"

KNOWLEDGE_BASE_DIR="$PROJECT/knowledge_base" \
RESULTS_DIR="$PROJECT/results" \
OLLAMA_BASE_URL="http://localhost:11434" \
nohup uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 \
    > "$LOGS/fastapi.log" 2>&1 &
disown

API_UP=false
for i in {1..20}; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        API_UP=true
        break
    fi
    sleep 2
done

if [ "$API_UP" != true ]; then
    echo " ✗ FastAPI failed to start. Last 30 lines of log:"
    echo "----------------------------------------------"
    tail -30 "$LOGS/fastapi.log"
    echo "----------------------------------------------"
    exit 1
fi

HEALTH=$(curl -s http://localhost:8000/health)
RAG_STATUS=$(echo "$HEALTH" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(f'RAG:{d.get(\"rag_ready\", False)} chunks:{d.get(\"chunks\", 0)}')
except Exception as e:
    print(f'RAG:parse_error({e})')
" 2>/dev/null)

echo " ✓ FastAPI ready — $RAG_STATUS"

# --- 5. Cloudflared Tunnel ---
echo "[5/5] Starting Cloudflared tunnel..."
pkill -f cloudflared 2>/dev/null || true
sleep 2

if [ ! -f "$STUDIO/cloudflared" ]; then
    echo " Downloading cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O "$STUDIO/cloudflared"
    chmod +x "$STUDIO/cloudflared"
fi

PUBLIC_URL=""
for attempt in 1 2 3; do
    echo " Tunnel attempt $attempt..."
    pkill -f cloudflared 2>/dev/null || true
    sleep 2
    > "$LOGS/cloudflared.log"
    nohup "$STUDIO/cloudflared" tunnel --url http://localhost:8000 > "$LOGS/cloudflared.log" 2>&1 &
    disown

    for i in {1..15}; do
        PUBLIC_URL=$(grep -o 'https://[^ ]*\.trycloudflare\.com' "$LOGS/cloudflared.log" 2>/dev/null | tail -1)
        [ -n "$PUBLIC_URL" ] && break
        sleep 2
    done

    if [ -n "$PUBLIC_URL" ]; then
        # verify it actually resolves end-to-end, not just printed
        sleep 3
        if curl -sf "$PUBLIC_URL/health" > /dev/null 2>&1; then
            break
        else
            echo " Tunnel URL printed but not yet resolving, retrying..."
            PUBLIC_URL=""
        fi
    fi
done

if [ -n "$PUBLIC_URL" ]; then
    echo "$PUBLIC_URL" > "$PROJECT/current_tunnel_url.txt"
    echo ""
    echo "=============================================="
    echo " 🚀 Project is READY!"
    echo "=============================================="
    echo "Models     : $MODEL_COUNT ($MODEL_NAMES)"
    echo "RAG Status : $RAG_STATUS"
    echo "Tunnel URL : $PUBLIC_URL"
    echo ""
    echo "Test health: curl -H 'x-api-key: pfe-anomaly-2025' $PUBLIC_URL/health"
    echo "Update your local env: LIGHTNING_URL=$PUBLIC_URL"
    echo "=============================================="
else
    echo " ✗ Could not establish a working tunnel after 3 attempts."
    echo " Check: cat $LOGS/cloudflared.log"
    exit 1
fi