# 🐾 MoodPet — Sistema Inteligente de Detecção Emocional

Bem-vindo ao MoodPet! Um sistema completo de identificação emocional em tempo real com pet virtual interativo e insights de IA.

## O que é MoodPet?

MoodPet é uma plataforma integrada de detecção contínua de expressões faciais que combina:
- **Análise emocional em tempo real** com modelos de Deep Learning
- **Pet virtual interativo** que reage às suas emoções
- **Insights comportamentais** gerados por IA
- **Alertas inteligentes** para emoções negativas prolongadas
- **Dashboard analítico** com histórico emocional

O sistema foi desenvolvido para facilitar o monitoramento do bem-estar emocional através de uma experiência lúdica e intuitiva.

## Objetivo

Criar uma rede de aplicações de bem-estar emocional que:
- Otimiza recursos através de uma arquitetura modular e reutilizável
- Reduz barreiras de entrada para implementação de detecção emocional
- Promove inovação em saúde mental e bem-estar digital
- Facilita integração entre diferentes plataformas (mobile, IoT)

## Histórico

A ideia do MoodPet nasceu da necessidade de criar ferramentas acessíveis para monitoramento de bem-estar emocional. Inspirado em plataformas de saúde digital e sistemas de IA empática, o projeto busca democratizar o acesso a tecnologias de detecção emocional.

## Modelo

MoodPet segue um modelo **modular e extensível**:
- **API Backend**: serviço centralizado de análise emocional
- **App Mobile**: interface intuitiva com pet virtual
- **Dispositivos IoT**: suporte para Raspberry Pi e outros dispositivos
- **Integração com IA**: Claude API para insights contextualizados

Cada componente pode funcionar independentemente ou integrado, permitindo diferentes formas de participação e uso.

## Formas de Participação

### Usuários Finais
- Utilizam o app mobile para monitoramento emocional
- Recebem alertas e insights personalizados
- Interagem com o pet virtual

### Desenvolvedores
- Integram a API em suas aplicações
- Desenvolvem novos componentes e extensões
- Contribuem com melhorias e novas funcionalidades

### Instituições
- Implementam soluções customizadas
- Integram com sistemas existentes
- Participam da evolução da plataforma

## Acesso aos Dados

Os dados do MoodPet são:
- **Privados por padrão**: armazenados localmente no dispositivo do usuário
- **Sincronizáveis**: opcionalmente sincronizados com backend
- **Auditáveis**: logs estruturados de todas as operações
- **Exportáveis**: dados podem ser exportados em formatos padrão

---

## Estrutura do Projeto

```
moodpet/
├── api/              # Backend FastAPI (Python)
│   ├── routers/      # Endpoints da API
│   ├── services/     # Lógica de negócio
│   ├── utils/        # Utilitários (DB, logging)
│   └── main.py       # Entry point
├── app/              # App React Native (Expo)
│   ├── src/
│   │   ├── screens/  # Telas da aplicação
│   │   ├── components/ # Componentes reutilizáveis
│   │   ├── hooks/    # Custom hooks
│   │   ├── services/ # Integração com API
│   │   └── theme/    # Design tokens
│   └── App.tsx       # Entry point
└── docs/
    └── iot_hardware.md   # Especificação de hardware IoT
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
- `POST /api/v1/emotion/analyze` — Análise de frames em tempo real
- `GET /api/v1/history/` — Histórico de eventos emocionais
- `GET /api/v1/history/summary` — Resumo por período
- `GET /api/v1/dashboard/overview` — Dashboard completo com IA
- `POST /api/v1/chat/` — Chat empático contextualizado
- `POST /api/v1/alerts/emotion` — Alertas de emoção negativa
- `POST /api/v1/alerts/no-face` — Alerta sem rosto detectado (10min)
- Banco SQLite com persistência de eventos
- Logs estruturados, tratamento de erros
- Documentação Swagger automática

### App Mobile
- Câmera rodando em background (invisível)
- Análise a cada 700ms sem interação do usuário
- Pet virtual animado (cachorro e gato) em SVG
- Animações por emoção: bounce, shake, pulse, float
- Badge com emoção + confiança + mensagem empática
- Sugestões musicais por emoção
- Dashboard com gráficos (PieChart + BarChart)
- Insight gerado por IA
- Chat empático (integrado com API)
- Configurações: nome/tipo do pet, contatos de emergência, alertas
- Estado global com Zustand
- Design system completo (branco gelo, tipografia, espaçamentos)

### IoT
- Especificação completa de hardware
- Lista de componentes com modelos recomendados
- Diagrama de conexões
- Descrição de design físico e experiência do usuário

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

## Tecnologia

### Stack Utilizado

| Camada | Stack |
|--------|-------|
| API | Python 3.11+, FastAPI, DeepFace, OpenCV, SQLite |
| App | React Native 0.81, TypeScript, Zustand, React Navigation, Expo |
| IA | Anthropic Claude API, OpenAI API |
| IoT | Raspberry Pi 5, Python, rpi_ws281x, pygame |

### Topologia de Componentes

**API Backend (Núcleo)**
- Responsável pelo processamento de frames e análise emocional
- Gerencia persistência de dados e histórico
- Coordena alertas e notificações
- Integra com serviços de IA

**App Mobile (Cliente)**
- Captura contínua de frames da câmera
- Interface intuitiva com pet virtual
- Sincronização com backend
- Notificações e alertas em tempo real

**Dispositivos IoT (Extensão)**
- Suporte para Raspberry Pi e similares
- Feedback visual (LEDs) baseado em emoções
- Integração com ecossistema smart home

---

## Governança

### Princípios de Desenvolvimento

- **Modularidade**: componentes independentes e reutilizáveis
- **Extensibilidade**: fácil integração de novos serviços
- **Qualidade**: testes automatizados e validação contínua
- **Documentação**: código bem documentado e exemplos práticos
- **Segurança**: proteção de dados e privacidade do usuário

### Decisões Técnicas

| Decisão | Justificativa |
|---------|---------------|
| FastAPI | Performance, async nativo, documentação automática |
| React Native + Expo | Cross-platform, desenvolvimento rápido, hot reload |
| SQLite | Leve, sem dependências externas, ideal para MVP |
| DeepFace | Modelos pré-treinados, alta precisão em detecção emocional |
| Zustand | State management simples e eficiente |
| Claude API | IA empática e contextualizada para insights |

### Roadmap

- [] Detecção de emoções em tempo real
- [] Pet virtual com animações
- [] Dashboard com histórico
- [] Chat empático com IA
- [] Alertas inteligentes
- [ ] Suporte a múltiplos idiomas
- [ ] Integração com wearables
- [ ] Análise preditiva de bem-estar
- [ ] Comunidade e compartilhamento de insights

---

## Qualidade e Segurança

### Padrões de Código

- **TypeScript strict mode** no frontend
- **Type hints** em todo código Python
- **Linting automático** com ESLint e Pylint
- **Formatação** com Prettier e Black
- **Testes unitários** para lógica crítica

### Segurança

- Validação de entrada em todos os endpoints
- Proteção contra CORS misconfiguration
- Logs estruturados sem exposição de dados sensíveis
- Suporte a variáveis de ambiente para configurações sensíveis
- Health checks e monitoramento de disponibilidade

### Observabilidade

- Logs estruturados em JSON
- Rastreamento de requisições com correlation IDs
- Métricas de performance
- Alertas para anomalias

---


## Créditos
Endrigo Gustavo Brandão de Oliveira
Gabriel Messias da Silva
Pedro Fernandes Araújo
