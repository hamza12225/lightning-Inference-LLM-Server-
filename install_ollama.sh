#!/bin/bash
GREEN='\033[0;32m'; NC='\033[0m'
log() { echo -e "${GREEN}[OLLAMA]${NC} $1"; }

log "Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

log "Starting Ollama server..."
pkill -f "ollama serve" 2>/dev/null || true
nohup ollama serve > logs/ollama.log 2>&1 &
sleep 8

log "Pulling models (this may take time)..."
ollama pull mistral:7b
ollama pull phi4-mini:latest

ollama list
log "Ollama + models ready ✅"
