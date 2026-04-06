# 🐾 MoodPet — App Expo

Aplicativo mobile para detecção de emoções em tempo real com pet virtual interativo.

## Pré-requisitos

- Node.js >= 18
- Expo CLI via `npx`
- Android Studio (para Android) ou um dispositivo com Expo Go
- JDK 17

## Quickstart

```bash
# 1. Instale as dependências
cd app
npm install

# 2. Inicie o Expo
npm start

# 3. (Android) Abra no emulador/dispositivo
npm run android

# 4. (iOS) Inicie o Expo no macOS
npm run ios
```

Para desenvolvimento com Expo Go no Android, também funciona:

```bash
npx expo start
```

## Permissões necessárias

**Android** (`AndroidManifest.xml` — já configurado):
- `CAMERA`
- `RECORD_AUDIO`
- `INTERNET`
- `VIBRATE`

**iOS** (`Info.plist` — já configurado):
- `NSCameraUsageDescription`

## Arquitetura

```
app/
├── App.tsx                          # Entry point
├── src/
│   ├── screens/
│   │   ├── HomeScreen.tsx           # 🐾 Tela principal com pet
│   │   ├── DashboardScreen.tsx      # 📊 Dashboard + insights IA
│   │   ├── ChatScreen.tsx           # 💬 Chat empático
│   │   └── SettingsScreen.tsx       # ⚙️ Configurações
│   ├── components/
│   │   ├── VirtualPet.tsx           # Pet SVG animado
│   │   └── EmotionBadge.tsx         # Badge com emoção + confiança
│   ├── hooks/
│   │   ├── useStore.ts              # Estado global (Zustand)
│   │   └── useEmotionDetection.ts   # Loop de captura + análise
│   ├── services/
│   │   └── api.ts                   # Cliente HTTP para a API
│   ├── navigation/
│   │   └── AppNavigator.tsx         # Stack navigator
│   └── theme/
│       └── index.ts                 # Design tokens
```

## Fluxo de funcionamento

1. App abre → solicita permissão de câmera
2. Câmera frontal ativa em background (invisível para o usuário)
3. A cada 700ms → captura snapshot → envia para API
4. API retorna emoção → pet reage com animação
5. Badge exibe emoção + confiança + mensagem empática
6. Se emoção negativa prolongada → dispara alerta para contatos
7. Se sem rosto por X min → alerta no-face

## Configuração da API

Edite `src/services/api.ts`:
```typescript
const BASE_URL = 'http://SEU_IP:8000'; // Substitua pelo IP da API
```

Em emulador Android: use `http://10.0.2.2:8000`
Em dispositivo físico: use o IP local da máquina
