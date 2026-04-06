# 🐾 MoodPet — Sistema Completo de Identificação Emocional

> Detecção contínua de expressões faciais + pet virtual interativo + insights de IA

---

## Estrutura do Projeto

```
moodpet/
├── api/              # Backend FastAPI (Python)
├── app/              # App React Native
└── docs/
    └── iot_hardware.md   # Especificação completa do dispositivo IoT
```

---

## Como Rodar (Passo a Passo)

### 1. API Backend

```bash
cd api

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate       # Linux/macOS
# ou: venv\Scripts\activate    # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Edite .env e adicione sua ANTHROPIC_API_KEY

# Rodar
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Acesse:
- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/health

---

### 2. App Mobile

```bash
cd app

# Instalar dependências
npm install

# Configurar IP da API em src/services/api.ts
# Para emulador Android: http://10.0.2.2:8000
# Para dispositivo físico: http://SEU_IP_LOCAL:8000

# Rodar
npx react-native run-android
# ou
npx react-native run-ios
```

---

## Funcionalidades Implementadas

### API
- ✅ `POST /api/v1/emotion/analyze` — Análise de frames em tempo real
- ✅ `GET /api/v1/history/` — Histórico de eventos emocionais
- ✅ `GET /api/v1/history/summary` — Resumo por período
- ✅ `GET /api/v1/dashboard/overview` — Dashboard completo com IA
- ✅ `POST /api/v1/chat/` — Chat empático contextualizado
- ✅ `POST /api/v1/alerts/emotion` — Alertas de emoção negativa
- ✅ `POST /api/v1/alerts/no-face` — Alerta sem rosto detectado (10min)
- ✅ Banco SQLite com persistência de eventos
- ✅ Logs estruturados, tratamento de erros
- ✅ Documentação Swagger automática

### App Mobile
- ✅ Câmera rodando em background (invisível)
- ✅ Análise a cada 700ms sem interação do usuário
- ✅ Pet virtual animado (cachorro e gato) em SVG
- ✅ Animações por emoção: bounce, shake, pulse, float
- ✅ Badge com emoção + confiança + mensagem empática
- ✅ Sugestões musicais por emoção
- ✅ Dashboard com gráficos (PieChart + BarChart)
- ✅ Insight gerado por IA
- ✅ Chat empático (integrado com API)
- ✅ Configurações: nome/tipo do pet, contatos de emergência, alertas
- ✅ Estado global com Zustand
- ✅ Design system completo (branco gelo, tipografia, espaçamentos)

### IoT
- ✅ Especificação completa de hardware
- ✅ Lista de componentes com modelos recomendados
- ✅ Diagrama de conexões
- ✅ Descrição de design físico e experiência do usuário

---

## Arquitetura

```
┌──────────────┐     frame (base64)     ┌──────────────┐
│  App Mobile  │ ──────────────────────▶│  FastAPI     │
│  React Native│ ◀─────────────────────│  (Python)    │
└──────────────┘   emotion + message    │              │
                                        │  DeepFace /  │
┌──────────────┐     frame (base64)     │  OpenCV      │
│  IoT Device  │ ──────────────────────▶│              │
│  Raspberry Pi│ ◀─────────────────────│  Anthropic   │
└──────────────┘   emotion + LEDs       │  Claude API  │
                                        └──────────────┘
                                               │
                                         SQLite DB
```

---

## Configurações de IA (Opcional mas Recomendado)

O sistema funciona sem IA configurada (modo fallback), mas para:
- **Chat empático contextualizado**: adicione `ANTHROPIC_API_KEY` no `.env`
- **Insights do dashboard**: mesma chave acima

---

## Tecnologias

| Camada | Stack |
|--------|-------|
| API | Python 3.11, FastAPI, DeepFace, OpenCV, SQLite |
| App | React Native 0.74, TypeScript, Zustand, React Navigation |
| IA | Anthropic Claude API |
| IoT | Raspberry Pi 5, Python, rpi_ws281x, pygame |
