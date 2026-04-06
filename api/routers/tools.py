"""
Tools Router — Emotion-aware quick actions and recovery plans
"""

from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from utils.logger import setup_logger

router = APIRouter()
logger = setup_logger(__name__)


class ToolItem(BaseModel):
    id: str
    title: str
    subtitle: str
    icon: str
    action: str
    accent: bool = False


class ToolsResponse(BaseModel):
    emotion: str
    confidence: float
    support_message: str
    suggested_tools: list[ToolItem]
    micro_plan: list[str]
    focus_mode_hint: str


class BreathingProtocolResponse(BaseModel):
    emotion: str
    cycle_seconds: int
    inhale_seconds: int
    hold_seconds: int
    exhale_seconds: int
    hold_after_exhale_seconds: int
    rounds: int
    intro: str


class DailyContentResponse(BaseModel):
    emotion: str
    affirmation: str
    journal_prompt: str
    reset_prompt: str


def _build_daily_content(emotion: str) -> DailyContentResponse:
    catalog = {
        "happy": {
            "affirmation": "Eu valorizo esse momento bom e posso usá-lo com intenção.",
            "journal_prompt": "Que pequena vitória de hoje merece ser registrada?",
            "reset_prompt": "Qual próxima ação simples mantém essa energia viva?",
        },
        "sad": {
            "affirmation": "Eu posso sentir tudo isso e ainda dar um passo pequeno e gentil.",
            "journal_prompt": "O que mais está pesando agora, sem filtro?",
            "reset_prompt": "Qual ação mínima me ajudaria nos próximos 10 minutos?",
        },
        "anxious": {
            "affirmation": "Eu volto para o agora, um ciclo de respiração por vez.",
            "journal_prompt": "Qual medo está mais alto agora e o que é fato sobre ele?",
            "reset_prompt": "O que está no meu controle imediato neste instante?",
        },
        "fearful": {
            "affirmation": "Eu posso buscar segurança em passos curtos e claros.",
            "journal_prompt": "O que eu preciso para me sentir 10% mais seguro agora?",
            "reset_prompt": "Quem ou o que pode me dar suporte prático hoje?",
        },
        "angry": {
            "affirmation": "Eu posso pausar, descarregar tensão e escolher com clareza.",
            "journal_prompt": "O que cruzou meu limite e como quero comunicar isso depois?",
            "reset_prompt": "O que evita que eu responda no impulso agora?",
        },
        "neutral": {
            "affirmation": "Consistência pequena também é progresso real.",
            "journal_prompt": "Qual tarefa curta merece prioridade agora?",
            "reset_prompt": "Que passo de 5 minutos eu posso concluir já?",
        },
    }
    selected = catalog.get(emotion, catalog["neutral"])
    return DailyContentResponse(emotion=emotion, **selected)


def _build_breathing_protocol(emotion: str, confidence: float) -> BreathingProtocolResponse:
    if emotion in {"sad", "anxious", "fearful"}:
        rounds = 7 if confidence >= 0.7 else 5
        return BreathingProtocolResponse(
            emotion=emotion,
            cycle_seconds=16,
            inhale_seconds=4,
            hold_seconds=4,
            exhale_seconds=4,
            hold_after_exhale_seconds=4,
            rounds=rounds,
            intro="Respiração em caixa para reduzir ativação e recuperar foco.",
        )
    if emotion == "angry":
        rounds = 6 if confidence >= 0.7 else 4
        return BreathingProtocolResponse(
            emotion=emotion,
            cycle_seconds=14,
            inhale_seconds=4,
            hold_seconds=2,
            exhale_seconds=6,
            hold_after_exhale_seconds=2,
            rounds=rounds,
            intro="Expiração mais longa para ajudar a baixar a intensidade.",
        )

    return BreathingProtocolResponse(
        emotion=emotion,
        cycle_seconds=12,
        inhale_seconds=4,
        hold_seconds=2,
        exhale_seconds=4,
        hold_after_exhale_seconds=2,
        rounds=4,
        intro="Respiração de manutenção para estabilizar presença e energia.",
    )


def _build_tools(emotion: str, confidence: float, face_detected: bool) -> ToolsResponse:
    confident = confidence >= 0.65

    if emotion == "happy":
        tools = [
            ToolItem(id="celebrate", title="Registrar conquista", subtitle="Guardar o que deu certo hoje", icon="star-check-outline", action="open_journal", accent=True),
            ToolItem(id="share", title="Compartilhar energia", subtitle="Mandar uma mensagem boa para alguém", icon="share-variant-outline", action="open_chat"),
            ToolItem(id="keep_flow", title="Manter o ritmo", subtitle="Um plano leve para as próximas 2h", icon="clock-fast", action="open_focus"),
            ToolItem(id="music", title="Trilha positiva", subtitle="Ouvir algo que combine com esse momento", icon="music-note-outline", action="open_music"),
            ToolItem(id="hydrate", title="Pausa de água", subtitle="Hidratar e manter energia", icon="cup-water", action="hydrate"),
            ToolItem(id="stretch", title="Alongamento rápido", subtitle="2 minutos para soltar o corpo", icon="human-handsup", action="start_stretch"),
        ]
        plan = ["Aproveite esse momento por 2 minutos", "Registre uma vitória pequena", "Envie uma mensagem boa para alguém"]
        support = "Você está com uma energia boa. Vale transformar isso em algo concreto."
        hint = "Use o modo foco leve para continuar sem perder o embalo."
    elif emotion in {"sad", "anxious", "fearful"}:
        tools = [
            ToolItem(id="breath", title="Respiração guiada", subtitle="Desacelerar o corpo em 1 minuto", icon="meditation", action="start_breathing", accent=True),
            ToolItem(id="ground", title="Aterramento 5-4-3-2-1", subtitle="Voltar para o presente agora", icon="earth", action="grounding"),
            ToolItem(id="checkin", title="Check-in rápido", subtitle="Dizer em uma frase como você está", icon="clipboard-text-outline", action="open_chat"),
            ToolItem(id="support", title="Plano de apoio", subtitle="Ver contatos e próximos passos", icon="account-heart-outline", action="open_support"),
            ToolItem(id="safe_space", title="Criar espaço seguro", subtitle="Silenciar estímulos por 5 minutos", icon="weather-night", action="open_reset"),
            ToolItem(id="self_compassion", title="Autocompaixão", subtitle="Uma frase para reduzir pressão", icon="heart-outline", action="self_compassion"),
        ]
        plan = ["Respire por 60 segundos", "Fale uma frase sobre o que pesa", "Escolha uma ação pequena e segura"]
        support = "Vou priorizar calma, clareza e passos pequenos com você."
        hint = "Se a sensação estiver alta, use o modo foco para reduzir estímulos."
    elif emotion == "angry":
        tools = [
            ToolItem(id="pause", title="Pausa curta", subtitle="Desarmar o impulso por 30 segundos", icon="pause-circle-outline", action="open_pause", accent=True),
            ToolItem(id="breath", title="Respiração de descarga", subtitle="Soltar a tensão com o corpo", icon="lungs", action="start_breathing"),
            ToolItem(id="note", title="Escrever sem filtro", subtitle="Descarregar em texto antes de agir", icon="pencil-outline", action="open_chat"),
            ToolItem(id="distance", title="Criar distância", subtitle="Reduzir gatilhos por alguns minutos", icon="step-backward", action="open_focus"),
            ToolItem(id="cold_water", title="Água fria no rosto", subtitle="Queda rápida de ativação", icon="snowflake", action="cooldown"),
            ToolItem(id="body_reset", title="Reset corporal", subtitle="20 agachamentos ou caminhada curta", icon="run-fast", action="start_stretch"),
        ]
        plan = ["Não responda no impulso", "Afaste-se do gatilho por 2 minutos", "Escreva o que você queria dizer"]
        support = "Seu corpo está pedindo descarga. Vamos baixar a intensidade antes de decidir qualquer coisa."
        hint = "Um tempo curto fora do gatilho costuma ajudar muito."
    else:
        tools = [
            ToolItem(id="focus", title="Modo foco", subtitle="Organizar o próximo passo", icon="target", action="open_focus", accent=True),
            ToolItem(id="journal", title="Diário rápido", subtitle="Capturar o que está na cabeça", icon="notebook-outline", action="open_chat"),
            ToolItem(id="breath", title="Respiração guiada", subtitle="Preparar o corpo para continuar", icon="meditation", action="start_breathing"),
            ToolItem(id="summary", title="Resumo do humor", subtitle="Ver padrões da semana", icon="chart-bar", action="open_dashboard"),
            ToolItem(id="reset", title="Reset de 3 minutos", subtitle="Pausa curta com guia", icon="timer-sand", action="open_reset"),
            ToolItem(id="hydrate", title="Pausa de água", subtitle="Recuperar foco rapidamente", icon="cup-water", action="hydrate"),
        ]
        plan = ["Escolha uma prioridade pequena", "Registre o que precisa ser lembrado", "Dê um passo de cada vez"]
        support = "Você não precisa resolver tudo agora."
        hint = "Use o modo foco para transformar a próxima ação em algo simples."

    if not face_detected:
        tools.insert(0, ToolItem(id="camera", title="Ajustar rosto", subtitle="Reconectar a leitura emocional", icon="camera-retake-outline", action="open_home", accent=True))
        plan = ["Ajuste a câmera e o rosto no quadro", "Espere uma nova leitura", *plan]
        support = "Sem rosto detectado, eu deixei uma ação de reconexão no topo."

    if confident:
        plan.append("Se quiser, siga para o chat com uma frase curta")

    return ToolsResponse(
        emotion=emotion,
        confidence=confidence,
        support_message=support,
        suggested_tools=tools,
        micro_plan=plan[:4],
        focus_mode_hint=hint,
    )


@router.get("/recommendations", response_model=ToolsResponse, summary="Emotion aware tool recommendations")
async def get_tool_recommendations(
    emotion: str = Query("neutral"),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
    face_detected: bool = Query(True),
):
    logger.info(
        "tools recommendations | emotion=%s confidence=%.3f face_detected=%s",
        emotion,
        confidence,
        face_detected,
    )
    return _build_tools(emotion, confidence, face_detected)


@router.get("/daily-content", response_model=DailyContentResponse, summary="Daily affirmation and prompts by emotion")
async def get_daily_content(emotion: str = Query("neutral")):
    logger.info("tools daily-content | emotion=%s", emotion)
    return _build_daily_content(emotion)


@router.get("/breathing-protocol", response_model=BreathingProtocolResponse, summary="Breathing protocol by emotion")
async def get_breathing_protocol(
    emotion: str = Query("neutral"),
    confidence: float = Query(0.5, ge=0.0, le=1.0),
):
    logger.info("tools breathing-protocol | emotion=%s confidence=%.3f", emotion, confidence)
    return _build_breathing_protocol(emotion, confidence)