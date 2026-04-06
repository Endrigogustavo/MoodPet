# 🤖 MoodPet IoT — Especificação Completa do Hardware

## Visão Geral

O **MoodPet IoT** é um dispositivo físico complementar ao app mobile. Ele se apresenta como um cubo com bordas completamente arredondadas (estilo pebble), com design fofo e amigável, capaz de detectar emoções de forma autônoma, exibir o pet virtual em sua tela frontal e interagir com o usuário de forma empática através de luz, som e vibração.

---

## Design Físico

### Forma e Material
- **Geometria**: Cubo com bordas super-arredondadas (raio de curvatura ≈ 3.5cm), similar a um marshmallow ou pebble
- **Dimensões**: ~12cm × 12cm × 12cm
- **Material externo**: Polímero ABS macio ao toque, acabamento fosco
- **Cor padrão**: Branco neve (com versão grafite e lavanda)
- **Peso estimado**: ~450g (com bateria)

### Design de Superfície
- **Face frontal**: Tela AMOLED redonda de 3.5" embutida na face, levemente recuada (como um "rosto" afundado no cubo)
- **Face superior**: Botão físico suave e arredondado (ação: acordar/modo silencioso)
- **Face inferior**: Base antiderrapante com emborrachamento + entrada USB-C para carga
- **Laterais**: Sem elementos visuais — superfície lisa e continua
- **Face posterior**: Grade de ventilação + câmera discreta flush-mounted no canto superior esquerdo

### Iluminação
- Anel de LED RGB difuso ao redor da tela frontal (tipo "aura")
- LEDs de borda nas quinas superiores (4 pontos de luz suave)
- A iluminação pulsa e muda de cor conforme a emoção detectada

### Feedback emocional visual da caixa
| Emoção     | Cor da aura | Animação         |
|------------|-------------|------------------|
| Feliz      | Amarelo ouro | Pulso vivo       |
| Triste     | Azul suave  | Pulso lento       |
| Raiva      | Vermelho    | Tremor rápido    |
| Ansioso    | Roxo suave  | Respiração lenta |
| Neutro     | Branco/prata| Fade suave       |
| Surpreso   | Rosa quente | Flash rápido     |

---

## Componentes de Hardware

### Processador Principal

| Componente | Modelo Recomendado | Justificativa |
|------------|-------------------|---------------|
| **SBC (Single Board Computer)** | Raspberry Pi 5 (4GB RAM) | Poder suficiente para ML embarcado + I/O em tempo real |
| **Alternativa** | Raspberry Pi 4 Model B (4GB) | Mais barato, ligeiramente mais lento para inferência |

**Especificações do Raspberry Pi 5:**
- Processador Quad-Core ARM Cortex-A76 a 2.4GHz
- 4GB LPDDR4X-4267 SDRAM
- Suporte a câmera CSI × 2
- PCIe 2.0 para módulos de aceleração
- Consumo: ~5W idle, ~12W load
- Forma: 85mm × 56mm

---

### Display (Rosto do Pet)

| Componente | Modelo Recomendado |
|------------|-------------------|
| **Display** | Waveshare 3.5" Round LCD Display (480×480) |
| **Interface** | SPI ou HDMI via adaptador |
| **Alternativa** | Waveshare 4" Round Touch Display (720×720) — mais nítido |

**Detalhes:**
- Tela circular AMOLED, 480×480 px
- Taxa de atualização 60Hz
- Ângulo de visão 170°
- Conecta via GPIO SPI no RPi
- Exibirá o pet animado (renderizado pelo app web local ou pipeline Python)

---

### Câmera

| Componente | Modelo Recomendado |
|------------|-------------------|
| **Câmera** | Raspberry Pi Camera Module 3 (12MP, wide) |
| **Alternativa** | Arducam IMX519 16MP Auto-focus |

**Especificações:**
- Resolução: 12MP (4608×2592)
- FPS a 1080p30: suficiente para análise em 30fps
- HDR nativo
- Conexão: cabo flat CSI-2 de 15 pinos
- Campo de visão: 102° (versão wide)
- Instalação: montada discretamente na face posterior superior ou em miniencaixe na face frontal abaixo da tela

---

### Armazenamento

| Componente | Modelo Recomendado |
|------------|-------------------|
| **SD Card** | SanDisk Extreme 64GB U3 A2 |
| **Backup/Expansão** | SSD NVMe M.2 2230 via HAT (opcional) |

---

### Áudio (Alto-falante e Microfone)

| Componente | Modelo Recomendado |
|------------|-------------------|
| **DAC + Amp** | Pimoroni Speaker pHAT (I2S) |
| **Alternativa** | Adafruit I2S 3W Stereo Speaker Bonnet |
| **Alto-falante** | 2× Full-range 3W 4Ω, 40mm, raso |
| **Microfone** | Adafruit PDM MEMS Microphone (SPH0645) |

**Posicionamento:**
- Alto-falantes embutidos nas laterais esquerd/direita (grade discreta)
- Microfone no topo, ao lado do botão físico

---

### Vibração

| Componente | Modelo Recomendado |
|------------|-------------------|
| **Motor** | LRA (Linear Resonant Actuator) 10mm × 3.4mm via DRV2605L I2C driver |
| **Alternativa** | ERM Motor 10mm C-type com transistor NPN |

**Funcionalidade:**
- Vibração padrão: notificação suave (single tap)
- Vibração de alerta: padrão rítmico (SOS-like)
- Vibração de felicidade: pulso duplo rápido

---

### Conectividade

| Componente | Modelo/Tipo |
|------------|-------------|
| **Wi-Fi** | Integrado RPi 5 — 802.11ac dual-band |
| **Bluetooth** | Integrado RPi 5 — BT 5.0 |
| **USB** | 2× USB 3.0 + 2× USB 2.0 (RPi 5) |
| **Ethernet** | RJ-45 Gigabit (RPi 5) |

---

### Energia

| Componente | Modelo Recomendado |
|------------|-------------------|
| **Carregamento** | Entrada USB-C PD (5V/5A = 25W) |
| **Bateria** | Li-Po 10.000mAh 3.7V × 2 em paralelo (portabilidade) |
| **UPS HAT** | Waveshare UPS HAT (C) — gerencia carga/descarga + indicador |
| **Regulador** | Integrado no UPS HAT (step-up para 5V) |

**Autonomia estimada:** ~4-6h de operação contínua com câmera ativa

---

### LEDs de Aura (Iluminação Emocional)

| Componente | Modelo |
|------------|--------|
| **LED Ring** | Adafruit NeoPixel Ring 24-LED (RGB WS2812B) |
| **LEDs de quina** | 4× NeoPixel Individual (WS2812B 5mm) |
| **Driver** | GPIO via `rpi_ws281x` Python library |

---

### Sensores Adicionais (Opcionais v2)

| Sensor | Modelo | Uso |
|--------|--------|-----|
| Temperatura/Umidade | DHT22 | Ambiente — contexto emocional |
| Luz ambiente | BH1750 I2C | Ajuste de brilho automático |
| Presença PIR | HC-SR501 | Detectar proximidade do usuário |
| Toque capacitivo | MPR121 I2C | Interação por toque no cubo |

---

## Diagrama de Conexões (Resumo)

```
┌─────────────────────────────────────────────────────┐
│                  Raspberry Pi 5                      │
│                                                     │
│  CSI ──── Camera Module 3                           │
│  SPI ──── Display Round 3.5"                        │
│  I2S ──── Speaker pHAT ──── 2× Alto-falantes        │
│  I2C ──── DRV2605L ──── LRA Vibration Motor         │
│  I2C ──── MPR121 (touch)                            │
│  GPIO 18 ─ NeoPixel LED Ring (aura) + corner LEDs   │
│  GPIO 17 ─ Botão físico (top)                       │
│  GPIO 27 ─ Microfone PDM                            │
│  I2C ──── UPS HAT ──── Li-Po Battery × 2           │
│  USB-C ── Entrada de carga externa                  │
└─────────────────────────────────────────────────────┘
```

---

## Stack de Software Embarcado

```
OS: Raspberry Pi OS Lite (64-bit, Bookworm)
Runtime: Python 3.11

Serviços:
├── camera_service.py        # Captura contínua de frames
├── emotion_client.py        # Envia frames para API MoodPet
├── display_service.py       # Renderiza pet na tela round
├── audio_service.py         # Controle de som (pygame/aplay)
├── led_service.py           # Anima LEDs conforme emoção
├── vibration_service.py     # Padrões hápticos
└── main.py                  # Orquestrador central

Comunicação: HTTPS → API MoodPet (mesma API do app mobile)
Configuração: /etc/moodpet/config.json (Wi-Fi, API URL, user_id)
Autostart: systemd service (moodpet.service)
```

---

## Lista de Compras Completa

| Item | Modelo | Qtd | Preço Est. (USD) |
|------|--------|-----|-----------------|
| Raspberry Pi 5 4GB | SC1111 | 1 | $60 |
| RPi Camera Module 3 Wide | SC0872 | 1 | $35 |
| Round Display 3.5" 480×480 | Waveshare | 1 | $38 |
| UPS HAT (C) | Waveshare | 1 | $25 |
| Li-Po 5000mAh 3.7V | Generic | 2 | $20 |
| Speaker pHAT | Pimoroni | 1 | $22 |
| Alto-falante 3W 4Ω 40mm | Generic | 2 | $8 |
| DRV2605L HAT | Adafruit #2305 | 1 | $8 |
| LRA Motor 10mm | Generic | 1 | $5 |
| NeoPixel Ring 24-LED | Adafruit #1586 | 1 | $16 |
| NeoPixel Individual 5mm | Adafruit | 4 | $8 |
| Microfone PDM SPH0645 | Adafruit #3421 | 1 | $6 |
| SD Card 64GB U3 A2 | SanDisk | 1 | $15 |
| USB-C PD Breakout | Adafruit | 1 | $4 |
| Carcaça impressa 3D | ABS/PLA matte white | 1 | $30–80 |
| **Total estimado** | | | **~$300–350** |

---

## Processo de Fabricação da Carcaça

1. **Arquivo 3D**: Modelar no Fusion 360 ou Blender (arquivo `.STL` / `.STEP`)
2. **Impressão**: FDM em ABS (resistente ao calor) ou resina SLA para acabamento superior
3. **Acabamento**: Lixamento progressivo (220→400→800 grit) + primer + tinta fosca branca
4. **Montagem**: Tampa posterior magnética para acesso ao RPi
5. **Encaixes**: Trilhos internos para fixar o RPi, bateria e HATs sem parafusos visíveis

---

## Experiência do Usuário (UX Física)

- **Ligar**: Pressionar botão superior por 2s → LED ring acende branco, tela mostra animação de boot com logo MoodPet
- **Em uso**: Tela exibe pet animado; LEDs de aura pulsam conforme emoção; nenhuma interação necessária
- **Notificação**: Vibração + LED amarelo piscante
- **Alerta**: Vibração forte + LED vermelho + áudio de alerta
- **Carregando**: LED azul pulsante suave
- **Modo silencioso**: Botão rápido → LEDs diminuem, som muda para haptics only
- **Posicionamento ideal**: Sobre mesa ou criado-mudo, câmera apontada para o rosto do usuário a ~50cm–120cm de distância
