"""
AI Chat Service — Empathic conversation engine powered by OpenAI or Anthropic
"""

import os
from typing import Optional

import anthropic
from dotenv import load_dotenv
from openai import OpenAI

from utils.logger import setup_logger

logger = setup_logger(__name__)
load_dotenv()

SYSTEM_PROMPT = """Você é o MoodPet, um assistente virtual empático e fofo que cuida do bem-estar emocional do usuário.

Você acabou de detectar a emoção do usuário através da câmera: {emotion} (confiança: {confidence}%).

Suas regras:
1. Seja SEMPRE gentil, acolhedor e não-julgador
2. Responda em português brasileiro, de forma natural e calorosa
3. Use emojis com moderação (1-2 por resposta)
4. Adapte o tom à emoção detectada:
   - Feliz: celebre junto, energia positiva
   - Triste: acolhimento, validação emocional, sugestão de autocuidado
   - Ansioso/com medo: técnicas de respiração, ancoragem no presente
   - Raiva: ajude a processar, sem minimizar
   - Neutro: leve, curioso, engajante
5. Respostas curtas (máximo 3 frases) e conversacionais
6. Use a emoção detectada só como contexto, sem soar como relatório técnico
7. Quando fizer sentido, faça uma pergunta curta de continuidade em vez de repetir o estado atual
8. Nunca dê diagnósticos médicos
9. Em casos graves, sugira ajuda profissional com carinho
10. Você tem personalidade de pet: fofo, leal, sempre presente
"""


class AIChatService:
    def __init__(self):
        self.provider = "fallback"
        self.openai_client = None
        self.anthropic_client = None

        openai_key = os.getenv("OPENAI_API_KEY", "")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")

        if openai_key:
            self.openai_client = OpenAI(api_key=openai_key)
            self.provider = "openai"
            self.available = True
            logger.info("✅ OpenAI client initialized")
        elif api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            self.provider = "anthropic"
            self.available = True
            logger.info("✅ Anthropic AI client initialized")
        else:
            self.available = False
            logger.warning("⚠️  OPENAI_API_KEY/ANTHROPIC_API_KEY not set — AI chat in fallback mode")

    async def generate_response(
        self,
        user_message: str,
        emotion: str,
        confidence: float,
        conversation_history: Optional[list] = None,
    ) -> tuple[str, str]:
        if not self.available:
            return self._fallback_response(emotion), self.provider

        system = SYSTEM_PROMPT.format(
            emotion=emotion,
            confidence=round(confidence * 100, 1),
        )

        messages = list(conversation_history or [])[-12:]
        messages.append({"role": "user", "content": user_message})

        try:
            if self.provider == "openai" and self.openai_client:
                formatted_messages = [{"role": "system", "content": system}] + messages
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=formatted_messages,
                    max_tokens=256,
                    temperature=0.8,
                )
                return response.choices[0].message.content or self._fallback_response(emotion), self.provider

            if self.provider == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=256,
                    system=system,
                    messages=messages,
                )
                return response.content[0].text, self.provider

            return self._fallback_response(emotion), self.provider
        except Exception as e:
            logger.error(f"AI chat error: {e}")
            return self._fallback_response(emotion), self.provider

    async def generate_insight(
        self,
        emotion_summary: dict,
        period: str = "hoje",
    ) -> tuple[str, str]:
        """Generate behavioral insight from emotion history."""
        if not self.available:
            return "Análise de padrões emocionais disponível com IA configurada.", self.provider

        prompt = f"""Analise o seguinte resumo emocional do usuário {period} e gere um insight empático, 
        construtivo e personalizado em 2-3 frases. Seja positivo e encorajador.
        
        Dados: {emotion_summary}
        
        Foque em: padrões observados, sugestões práticas, pontos positivos."""

        try:
            if self.provider == "openai" and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7,
                )
                return response.choices[0].message.content or "Continue acompanhando seu progresso emocional.", self.provider

            if self.provider == "anthropic" and self.anthropic_client:
                response = self.anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text, self.provider

            return "Continue acompanhando suas emoções — o autoconhecimento é o primeiro passo! 🌱", self.provider
        except Exception as e:
            logger.error(f"Insight generation error: {e}")
            return "Continue acompanhando suas emoções — o autoconhecimento é o primeiro passo! 🌱", self.provider

    def _fallback_response(self, emotion: str) -> str:
        fallbacks = {
            "happy": "Que energia incrível você está transmitindo! O MoodPet está adorando! 🌟",
            "sad": "Estou aqui com você. Momentos difíceis passam, e você não está sozinho(a) 💙",
            "angry": "Respira fundo comigo. Um passo de cada vez, tudo vai se resolver 🌿",
            "anxious": "Você está seguro(a) agora. Vamos respirar juntos: inspira 4s, expira 4s 💛",
            "neutral": "Como você está se sentindo hoje? Estou aqui para conversar! 🐾",
            "surprised": "Uau! O que aconteceu? Conta pra mim! 😲",
            "disgusted": "Caramba! Parece que algo não foi legal. Quer conversar? 💬",
        }
        return fallbacks.get(emotion, "Estou aqui com você 🐾")


ai_chat_service = AIChatService()
