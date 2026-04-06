import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Vibration,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import * as Speech from 'expo-speech';
import {
  ApiService,
  ToolsRecommendations,
  ToolItem as ApiToolItem,
  DailyContent,
  BreathingProtocol,
} from '../services/api';
import { useAppStore } from '../hooks/useStore';
import { Colors, Typography, Spacing, Radius, Shadow, EMOTION_COLORS } from '../theme';

const ToolCard = React.memo(function ToolCard({
  icon,
  title,
  subtitle,
  onPress,
  accent = false,
}: {
  icon: string;
  title: string;
  subtitle: string;
  onPress: () => void;
  accent?: boolean;
}) {
  return (
    <TouchableOpacity
      style={[styles.toolCard, accent && styles.toolCardAccent, Shadow.sm]}
      onPress={onPress}
      activeOpacity={0.88}
    >
      <View style={[styles.toolIconWrap, accent && styles.toolIconWrapAccent]}>
        <MaterialCommunityIcons
          name={icon as any}
          size={22}
          color={accent ? Colors.primary : Colors.textPrimary}
        />
      </View>
      <View style={styles.toolTextWrap}>
        <Text style={styles.toolTitle}>{title}</Text>
        <Text style={styles.toolSubtitle}>{subtitle}</Text>
      </View>
    </TouchableOpacity>
  );
});

export const ToolsScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const currentEmotion = useAppStore((state) => state.currentEmotion);
  const petName = useAppStore((state) => state.settings.petName);
  const voiceAssistantEnabled = useAppStore((state) => state.settings.voiceAssistantEnabled);
  const addChatMessage = useAppStore((state) => state.addChatMessage);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [recommendations, setRecommendations] = useState<ToolsRecommendations | null>(null);
  const [dailyContent, setDailyContent] = useState<DailyContent | null>(null);
  const [breathingProtocol, setBreathingProtocol] = useState<BreathingProtocol | null>(null);

  const emotionMeta = useMemo(
    () => EMOTION_COLORS[currentEmotion.emotion] || EMOTION_COLORS.neutral,
    [currentEmotion.emotion],
  );

  const openChatWithText = useCallback((text: string) => {
    addChatMessage({ role: 'assistant', content: text });
    navigation.navigate('Chat');
  }, [addChatMessage, navigation]);

  useEffect(() => {
    let cancelled = false;
    setLoadingRecommendations(true);
    ApiService.getToolRecommendations({
      emotion: currentEmotion.emotion,
      confidence: currentEmotion.confidence,
      faceDetected: currentEmotion.faceDetected,
    })
      .then((data) => {
        if (!cancelled) {
          setRecommendations(data);
        }
      })
      .catch((error) => {
        console.warn('[Tools] recommendation fetch error', error?.message || error);
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingRecommendations(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [currentEmotion.emotion, currentEmotion.confidence, currentEmotion.faceDetected]);

  useEffect(() => {
    let cancelled = false;

    ApiService.getDailyContent({ emotion: currentEmotion.emotion })
      .then((data) => {
        if (!cancelled) {
          setDailyContent(data);
        }
      })
      .catch((error) => {
        console.warn('[Tools] daily content fetch error', error?.message || error);
      });

    ApiService.getBreathingProtocol({
      emotion: currentEmotion.emotion,
      confidence: currentEmotion.confidence,
    })
      .then((data) => {
        if (!cancelled) {
          setBreathingProtocol(data);
        }
      })
      .catch((error) => {
        console.warn('[Tools] breathing protocol fetch error', error?.message || error);
      });

    return () => {
      cancelled = true;
    };
  }, [currentEmotion.emotion, currentEmotion.confidence]);

  const startBreathing = useCallback(() => {
    const inhale = breathingProtocol?.inhale_seconds ?? 4;
    const hold = breathingProtocol?.hold_seconds ?? 4;
    const exhale = breathingProtocol?.exhale_seconds ?? 4;
    const rounds = breathingProtocol?.rounds ?? 5;

    addChatMessage({
      role: 'assistant',
      content: `Vamos para ${rounds} ciclos. Inspire ${inhale}s, segure ${hold}s, solte ${exhale}s e repita no seu ritmo.`,
    });
    Vibration.vibrate([0, 40, 40, 40]);
    if (voiceAssistantEnabled) {
      Speech.speak(`Vamos respirar juntos. Inspire ${inhale}, segure ${hold}, solte ${exhale}.`, {
        language: 'pt-BR',
        pitch: 1,
        rate: 0.96,
      });
    }
    navigation.navigate('Chat');
  }, [addChatMessage, breathingProtocol, navigation, voiceAssistantEnabled]);

  const grounding = useCallback(() => {
    addChatMessage({
      role: 'assistant',
      content: 'Aterramento rápido: procure 5 coisas que vê, 4 que toca, 3 que ouve, 2 que cheira e 1 que sente.',
    });
    Vibration.vibrate([0, 30, 30, 30]);
    if (voiceAssistantEnabled) {
      Speech.speak('Vamos fazer um aterramento rápido juntos.', {
        language: 'pt-BR',
        pitch: 1,
        rate: 0.96,
      });
    }
    navigation.navigate('Chat');
  }, [addChatMessage, navigation, voiceAssistantEnabled]);

  const journal = useCallback(() => {
    addChatMessage({
      role: 'assistant',
      content: dailyContent?.journal_prompt || 'Escreva uma frase sobre o que mais pesou hoje. Eu continuo com você daí.',
    });
    Vibration.vibrate(70);
    navigation.navigate('Chat');
  }, [addChatMessage, dailyContent?.journal_prompt, navigation]);

  const moodSummary = useCallback(() => {
    navigation.navigate('Dashboard');
  }, [navigation]);

  const openChat = useCallback(() => {
    navigation.navigate('Chat');
  }, [navigation]);

  const hydrate = useCallback(() => {
    Vibration.vibrate(50);
    openChatWithText('Vamos fazer uma pausa de água agora. Depois me diga em uma frase como seu corpo ficou.');
  }, [openChatWithText]);

  const startStretch = useCallback(() => {
    Vibration.vibrate([0, 30, 30, 30]);
    openChatWithText('Faça 2 minutos de alongamento leve (pescoço, ombros, costas) e me conte como ficou sua tensão.');
  }, [openChatWithText]);

  const openReset = useCallback(() => {
    openChatWithText('Reset rapido de 3 minutos: respire 4-4-6 por 6 ciclos, beba agua e escolha uma unica acao para agora.');
  }, [openChatWithText]);

  const selfCompassion = useCallback(() => {
    openChatWithText('Frase de autocompaixao: eu estou fazendo o melhor que posso com os recursos de hoje.');
  }, [openChatWithText]);

  const cooldown = useCallback(() => {
    openChatWithText('Passe agua fria no rosto por 20 segundos e volte para me dizer se a intensidade caiu.');
  }, [openChatWithText]);

  const handleToolAction = useCallback((action: string) => {
    switch (action) {
      case 'start_breathing':
        return startBreathing();
      case 'grounding':
        return grounding();
      case 'open_chat':
        return openChat();
      case 'open_dashboard':
        return moodSummary();
      case 'open_focus':
        return navigation.navigate('Home');
      case 'open_music':
        return navigation.navigate('Dashboard');
      case 'open_support':
        return openChatWithText('Vou te ajudar a montar um plano de apoio com calma. Me diga o que está mais pesado agora.');
      case 'open_pause':
        return openChatWithText('Vamos pausar por 30 segundos antes de decidir qualquer coisa.');
      case 'open_home':
        return navigation.navigate('Home');
      case 'open_journal':
        return journal();
      case 'hydrate':
        return hydrate();
      case 'start_stretch':
        return startStretch();
      case 'open_reset':
        return openReset();
      case 'self_compassion':
        return selfCompassion();
      case 'cooldown':
        return cooldown();
      default:
        return openChat();
    }
  }, [cooldown, grounding, hydrate, journal, moodSummary, navigation, openChat, openChatWithText, openReset, selfCompassion, startBreathing, startStretch]);

  return (
    <SafeAreaView style={styles.root} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.hero}>
          <View style={[styles.heroBadge, { backgroundColor: emotionMeta.light }]}>
            <MaterialCommunityIcons name="toolbox-outline" size={20} color={emotionMeta.bg} />
          </View>
          <Text style={styles.title}>Ferramentas</Text>
          <Text style={styles.subtitle}>
            {petName} deixa o próximo passo mais simples com ações rápidas, apoio e foco.
          </Text>
        </View>

        <View style={[styles.moodCard, Shadow.md]}>
          <Text style={styles.cardTitle}>Estado atual</Text>
          <View style={styles.moodRow}>
            <MaterialCommunityIcons name={emotionMeta.icon as any} size={28} color={emotionMeta.bg} />
            <View style={styles.moodTextWrap}>
              <Text style={styles.moodLabel}>{emotionMeta.label}</Text>
              <Text style={styles.moodHint}>
                {currentEmotion.emotionVariant} · {Math.round(currentEmotion.confidence * 100)}%
              </Text>
            </View>
          </View>
          <Text style={styles.moodSupport}>{currentEmotion.supportTip || 'Eu posso te ajudar a dar o próximo passo.'}</Text>
        </View>

        <View style={[styles.recommendCard, Shadow.sm]}>
          <View style={styles.recommendHeader}>
            <MaterialCommunityIcons name="auto-fix" size={18} color={Colors.primary} />
            <Text style={styles.cardTitle}>Ferramentas recomendadas</Text>
          </View>
          <Text style={styles.recommendText}>
            {loadingRecommendations
              ? 'Montando suas ações agora...'
              : recommendations?.support_message || 'Ações sugeridas a partir da sua emoção atual.'}
          </Text>
          {recommendations?.micro_plan?.length ? (
            <View style={styles.planWrap}>
              {recommendations.micro_plan.map((step, index) => (
                <View key={`${step}-${index}`} style={styles.planRow}>
                  <Text style={styles.planIndex}>{index + 1}</Text>
                  <Text style={styles.planText}>{step}</Text>
                </View>
              ))}
            </View>
          ) : null}
          <Text style={styles.focusHint}>{recommendations?.focus_mode_hint || 'O modo foco aparece aqui quando a emoção muda.'}</Text>
        </View>

        <View style={[styles.tipCard, Shadow.sm]}>
          <Text style={styles.cardTitle}>Conteudo diario</Text>
          <Text style={styles.tipText}>{dailyContent?.affirmation || 'Uma frase curta para te recentrar no presente.'}</Text>
          <TouchableOpacity
            style={styles.inlineAction}
            onPress={() => openChatWithText(dailyContent?.reset_prompt || 'Quero fazer um reset emocional leve agora.')}
            activeOpacity={0.88}
          >
            <MaterialCommunityIcons name="chat-processing-outline" size={16} color={Colors.primary} />
            <Text style={styles.inlineActionText}>Levar prompt para o chat</Text>
          </TouchableOpacity>
          {breathingProtocol ? (
            <Text style={styles.focusHint}>
              {breathingProtocol.intro} Ciclo: {breathingProtocol.inhale_seconds}-{breathingProtocol.hold_seconds}-{breathingProtocol.exhale_seconds}-{breathingProtocol.hold_after_exhale_seconds}s por {breathingProtocol.rounds} rounds.
            </Text>
          ) : null}
        </View>

        <View style={styles.grid}>
          {(recommendations?.suggested_tools?.length ? recommendations.suggested_tools : []).map((tool: ApiToolItem) => (
            <ToolCard
              key={tool.id}
              icon={tool.icon}
              title={tool.title}
              subtitle={tool.subtitle}
              onPress={() => handleToolAction(tool.action)}
              accent={tool.accent}
            />
          ))}
        </View>

        {!recommendations?.suggested_tools?.length ? (
          <View style={styles.fallbackCard}>
            <Text style={styles.cardTitle}>Ações rápidas</Text>
            <View style={styles.fallbackRow}>
              <ToolCard
                icon="meditation"
                title="Respiração guiada"
                subtitle="1 minuto para desacelerar"
                onPress={startBreathing}
                accent
              />
              <ToolCard
                icon="earth"
                title="Aterramento"
                subtitle="5-4-3-2-1 para voltar ao presente"
                onPress={grounding}
              />
            </View>
          </View>
        ) : null}

        <View style={[styles.tipCard, Shadow.sm]}>
          <Text style={styles.cardTitle}>Sugestão rápida</Text>
          <Text style={styles.tipText}>
            Se hoje estiver pesado, use uma ferramenta e depois mande só uma frase no chat. O resto eu organizo com você.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: Colors.background },
  scroll: { padding: Spacing.xl, gap: Spacing.base, paddingBottom: 48 },
  hero: {
    alignItems: 'center',
    paddingTop: 4,
    paddingBottom: Spacing.base,
  },
  heroBadge: {
    width: 52,
    height: 52,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  title: {
    fontSize: Typography.display1,
    fontWeight: Typography.extrabold,
    color: Colors.textPrimary,
    textAlign: 'center',
  },
  subtitle: {
    marginTop: 10,
    fontSize: Typography.base,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: 22,
  },
  moodCard: {
    backgroundColor: Colors.surface,
    borderRadius: Radius.xl,
    padding: Spacing.base,
    gap: 10,
  },
  cardTitle: {
    fontSize: Typography.base,
    fontWeight: Typography.semibold,
    color: Colors.textPrimary,
  },
  moodRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  moodTextWrap: {
    flex: 1,
    gap: 2,
  },
  moodLabel: {
    fontSize: Typography.base,
    fontWeight: Typography.semibold,
    color: Colors.textPrimary,
    textTransform: 'capitalize',
  },
  moodHint: {
    fontSize: Typography.sm,
    color: Colors.textTertiary,
  },
  moodSupport: {
    fontSize: Typography.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
  },
  fallbackCard: {
    backgroundColor: Colors.surface,
    borderRadius: Radius.xl,
    padding: Spacing.base,
    gap: 12,
  },
  fallbackRow: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
  },
  toolCard: {
    width: '48%',
    minHeight: 118,
    borderRadius: Radius.xl,
    backgroundColor: Colors.surface,
    padding: Spacing.base,
    gap: 10,
  },
  toolCardAccent: {
    backgroundColor: Colors.primaryLight,
  },
  toolIconWrap: {
    width: 38,
    height: 38,
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.background,
  },
  toolIconWrapAccent: {
    backgroundColor: 'rgba(255,255,255,0.75)',
  },
  toolTextWrap: {
    flex: 1,
    gap: 4,
  },
  toolTitle: {
    fontSize: Typography.sm,
    fontWeight: Typography.semibold,
    color: Colors.textPrimary,
  },
  toolSubtitle: {
    fontSize: Typography.xs,
    color: Colors.textTertiary,
    lineHeight: 16,
  },
  tipCard: {
    backgroundColor: Colors.surface,
    borderRadius: Radius.xl,
    padding: Spacing.base,
    gap: 8,
  },
  inlineAction: {
    marginTop: 4,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    alignSelf: 'flex-start',
    backgroundColor: Colors.primaryLight,
    borderRadius: Radius.md,
    paddingHorizontal: 10,
    paddingVertical: 8,
  },
  inlineActionText: {
    fontSize: Typography.xs,
    color: Colors.primary,
    fontWeight: Typography.semibold,
  },
  focusHint: {
    fontSize: Typography.xs,
    color: Colors.textTertiary,
    lineHeight: 16,
  },
  recommendCard: {
    backgroundColor: Colors.surface,
    borderRadius: Radius.xl,
    padding: Spacing.base,
    gap: 10,
  },
  recommendHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  recommendText: {
    fontSize: Typography.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  planWrap: {
    gap: 8,
  },
  planRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 10,
  },
  planIndex: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: Colors.primaryLight,
    color: Colors.primary,
    textAlign: 'center',
    lineHeight: 22,
    fontSize: Typography.xs,
    fontWeight: Typography.bold,
  },
  planText: {
    flex: 1,
    fontSize: Typography.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
  tipText: {
    fontSize: Typography.sm,
    color: Colors.textSecondary,
    lineHeight: 20,
  },
});
