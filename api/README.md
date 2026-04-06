# 🐾 MoodPet API

API backend para detecção contínua de expressões faciais e interação emocional inteligente.

## Stack

- **FastAPI** — framework web assíncrono
- **DeepFace** — detecção de emoções com deep learning
- **OpenCV** — processamento de imagem
- **SQLite** (via `databases` + `aiosqlite`) — persistência leve
- **Anthropic Claude** — IA empática para chat e insights

## Quickstart

```bash
# 1. Crie o ambiente virtual
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 2. Instale as dependências
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Configure as variáveis de ambiente
cp .env.example .env
# Edite api/.env com suas chaves (mínimo recomendado: OPENAI_API_KEY)

# 4. Rode a API
venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --loop asyncio --http h11 --reload
# ou: ./run_api.sh (carrega api/.env automaticamente; `MOODPET_RELOAD=0` para desativar reload)
```

Se aparecer erro de ambiente externo gerenciado (PEP 668), confirme que o terminal está com a venv ativada antes de instalar pacotes.

### Troubleshooting (Kali / PEP 668)

Se aparecer `externally-managed-environment` ou `ModuleNotFoundError: No module named 'cv2'`, você está usando o Python do sistema em vez da venv.

Use exatamente este fluxo:

```bash
cd api
source venv/bin/activate
python -m pip install -r requirements.txt
venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Se você ver `exit code 139` (segfault) ao subir com Uvicorn, use `--loop asyncio --http h11` e evite `uvicorn[standard]`.

A API estará disponível em: `http://localhost:8000`
Documentação Swagger: `http://localhost:8000/docs`

## Endpoints principais

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/api/v1/emotion/analyze` | Analisa um frame base64 |
| `GET`  | `/api/v1/history/` | Histórico de eventos emocionais |
| `GET`  | `/api/v1/history/summary` | Resumo agregado por período |
| `GET`  | `/api/v1/dashboard/overview` | Dashboard completo com IA |
| `POST` | `/api/v1/chat/` | Chat empático baseado na emoção |
| `POST` | `/api/v1/alerts/emotion` | Disparar alerta emocional |
| `POST` | `/api/v1/alerts/no-face` | Disparar alerta sem rosto |
| `GET`  | `/health` | Health check |

## Fluxo de uso

```
App → captura frame da câmera (background)
    → POST /api/v1/emotion/analyze (intervalo varia por plataforma; Android pode ser mais espaçado para evitar shutter sound)
    → recebe emotion + mensagem + sugestão musical
    → pet reage na tela
    → se emoção negativa por X tempo → POST /api/v1/alerts/emotion
    → se sem rosto por 10min → POST /api/v1/alerts/no-face
```

## Estrutura

```
api/
├── main.py              # Entry point FastAPI
├── requirements.txt
├── .env.example
├── routers/
│   ├── emotion.py       # Frame processing
│   ├── history.py       # Historical data
│   ├── alerts.py        # Alert management
│   ├── dashboard.py     # Analytics + AI insights
│   └── ai_chat.py       # Conversational AI
├── services/
│   ├── emotion_service.py  # DeepFace + OpenCV
│   ├── ai_service.py       # Anthropic integration
│   └── alert_service.py    # Email/SMS alerts
└── utils/
    ├── database.py      # SQLite + tables
    └── logger.py        # Structured logging
```

## Produção

Recomendação de deploy:

- Para máxima compatibilidade com DeepFace/TensorFlow, use Python 3.11/3.12.
- Em Python 3.13, **DeepFace é desativado** e o fallback OpenCV também é **desativado por padrão** por estabilidade (evita segfault/exit 139). Nesse modo, `/emotion/analyze` responde, mas tende a retornar `neutral` com `face_detected=false`.
- Para forçar o modo estável mesmo em Python 3.11/3.12, defina `MOODPET_DISABLE_OPENCV=1`.

Para deploy com Docker:

Use os arquivos `Dockerfile` e `docker-compose.yml` incluídos nesta pasta.

```bash
docker compose up --build
```
