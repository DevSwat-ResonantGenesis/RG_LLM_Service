# RG LLM Service

> **Part of the [ResonantGenesis](https://resonant.dev-swat.com) platform** — LLM HTTP gateway with context injection and tool execution.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)

## Features
- HTTP endpoint for chat completions
- Context injection from memory service
- Tool execution layer for agent modules
- Provider model list endpoint
- Multi-provider support (OpenAI, Anthropic, Groq, Gemini)

## Quick Start
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deployment
- **Container**: `llm_service` | **Port**: 8000
- **Server path**: `/home/deploy/RG_LLM_Service`

---
**Organization**: [DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) | **Platform**: [resonant.dev-swat.com](https://resonant.dev-swat.com)
