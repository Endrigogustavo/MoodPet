import axios, { AxiosInstance } from 'axios';
import Constants from 'expo-constants';

const resolveBaseUrl = () => {
  const envUrl = process.env.EXPO_PUBLIC_API_URL;
  if (envUrl) {
    return envUrl;
  }

  const apiPort = process.env.EXPO_PUBLIC_API_PORT || '8000';

  if (!__DEV__) {
    return `http://localhost:${apiPort}`;
  }

  const hostUri =
    (Constants as any)?.expoConfig?.hostUri ||
    (Constants as any)?.manifest2?.extra?.expoGo?.debuggerHost ||
    (Constants as any)?.manifest?.debuggerHost ||
    '';

  const host = typeof hostUri === 'string' ? hostUri.split(':')[0] : '';
  if (host) {
    return `http://${host}:${apiPort}`;
  }

  return `http://127.0.0.1:${apiPort}`;
};

const BASE_URL = resolveBaseUrl();

if (__DEV__) {
  console.log('[API] BASE_URL =', BASE_URL);
}

const api: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: 9000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Request interceptor ────────────────────────────────────────────────────────
api.interceptors.request.use(
  (config) => {
    if (__DEV__) {
      const method = (config.method || 'GET').toUpperCase();
      const url = `${config.baseURL || ''}${config.url || ''}`;
      console.log('[API] ->', method, url);
    }
    return config;
  },
  (error) => Promise.reject(error),
);

api.interceptors.response.use(
  (response) => {
    if (__DEV__) {
      const method = (response.config.method || 'GET').toUpperCase();
      const url = `${response.config.baseURL || ''}${response.config.url || ''}`;
      console.log('[API] <-', response.status, method, url);
    }
    return response;
  },
  (error) => {
    const method = (error?.config?.method || 'GET').toUpperCase();
    const url = `${error?.config?.baseURL || ''}${error?.config?.url || ''}`;
    const status = error?.response?.status;
    const data = error?.response?.data;
    const code = error?.code;
    const timeout = error?.config?.timeout;
    console.warn('[API] xx', status || 'NO_STATUS', method, url);

    // React Native / Axios often reports connection issues as "Network Error" with no status.
    // Include code/timeout to make it clear whether this is offline, refused, or timed out.
    if (data) {
      console.warn('[API Error]', data);
    } else {
      console.warn('[API Error]', {
        message: error?.message,
        code,
        timeout,
      });
    }
    return Promise.reject(error);
  },
);

// ── Types ──────────────────────────────────────────────────────────────────────

export interface EmotionAnalysisResult {
  emotion: string;
  emotion_variant: string;
  emotion_zone: string;
  support_tip: string;
  confidence: number;
  secondary_emotion: string | null;
  all_scores: Record<string, number>;
  face_detected: boolean;
  processing_time_ms: number;
  message: string | null;
  music_suggestions: string[];
  timestamp: number;
  no_face_alert: boolean;
  seconds_since_last_face: number;
  should_trigger_alert: boolean;
}

export interface EmotionSummary {
  period_hours: number;
  total_events: number;
  dominant_emotion: string;
  emotion_counts: Record<string, number>;
  emotion_percentages: Record<string, number>;
  avg_confidences: Record<string, number>;
  face_detected_count: number;
}

export interface DashboardData {
  period_hours: number;
  total_events: number;
  timeline: Array<{ hour: number; emotions: Record<string, number>; total: number }>;
  emotion_distribution: Record<string, number>;
  emotion_counts: Record<string, number>;
  wellbeing_score: number;
  peak_emotions: Array<[string, number]>;
  ai_insight: string;
  ai_provider?: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  response: string;
  emotion_context: string;
  provider: string;
}

export interface ToolItem {
  id: string;
  title: string;
  subtitle: string;
  icon: string;
  action: string;
  accent: boolean;
}

export interface ToolsRecommendations {
  emotion: string;
  confidence: number;
  support_message: string;
  suggested_tools: ToolItem[];
  micro_plan: string[];
  focus_mode_hint: string;
}

export interface DailyContent {
  emotion: string;
  affirmation: string;
  journal_prompt: string;
  reset_prompt: string;
}

export interface BreathingProtocol {
  emotion: string;
  cycle_seconds: number;
  inhale_seconds: number;
  hold_seconds: number;
  exhale_seconds: number;
  hold_after_exhale_seconds: number;
  rounds: number;
  intro: string;
}

// ── API Calls ──────────────────────────────────────────────────────────────────

export const ApiService = {
  // Health
  async health(options?: { timeoutMs?: number }) {
    const { data } = await api.get('/health', {
      timeout: options?.timeoutMs ?? 2500,
    });
    return data;
  },

  // Emotion
  async analyzeFrame(
    frameB64: string,
    sessionId: string,
    userId?: string,
    petName = 'MoodPet',
  ): Promise<EmotionAnalysisResult> {
    const { data } = await api.post('/api/v1/emotion/analyze', {
      frame_b64: frameB64,
      session_id: sessionId,
      user_id: userId,
      pet_name: petName,
      source: 'mobile',
    });
    return data;
  },

  // History
  async getHistory(params: { sessionId?: string; userId?: string; limit?: number }) {
    const { data } = await api.get('/api/v1/history/', { params: {
      session_id: params.sessionId,
      user_id: params.userId,
      limit: params.limit ?? 100,
    }});
    return data;
  },

  async getSummary(params: { userId?: string; sessionId?: string; hours?: number }): Promise<EmotionSummary> {
    const { data } = await api.get('/api/v1/history/summary', { params: {
      user_id: params.userId,
      session_id: params.sessionId,
      hours: params.hours ?? 24,
    }});
    return data;
  },

  // Dashboard
  async getDashboard(params: { userId?: string; sessionId?: string; hours?: number }): Promise<DashboardData> {
    const { data } = await api.get('/api/v1/dashboard/overview', { params: {
      user_id: params.userId,
      session_id: params.sessionId,
      hours: params.hours ?? 24,
    }});
    return data;
  },

  // Chat
  async chat(
    message: string,
    emotion: string,
    confidence: number,
    history: ChatMessage[] = [],
  ): Promise<ChatResponse> {
    const { data } = await api.post('/api/v1/chat/', {
      message,
      emotion,
      confidence,
      history: history.map((m) => ({ role: m.role, content: m.content })),
    });
    return data;
  },

  async getToolRecommendations(params: { emotion: string; confidence: number; faceDetected: boolean }): Promise<ToolsRecommendations> {
    const { data } = await api.get('/api/v1/tools/recommendations', {
      params: {
        emotion: params.emotion,
        confidence: params.confidence,
        face_detected: params.faceDetected,
      },
    });
    return data;
  },

  async getDailyContent(params: { emotion: string }): Promise<DailyContent> {
    const { data } = await api.get('/api/v1/tools/daily-content', {
      params: { emotion: params.emotion },
    });
    return data;
  },

  async getBreathingProtocol(params: { emotion: string; confidence: number }): Promise<BreathingProtocol> {
    const { data } = await api.get('/api/v1/tools/breathing-protocol', {
      params: {
        emotion: params.emotion,
        confidence: params.confidence,
      },
    });
    return data;
  },

  // Alerts
  async triggerEmotionAlert(payload: {
    session_id: string;
    user_name?: string;
    emotion: string;
    duration_minutes: number;
    contact_email?: string;
  }) {
    const { data } = await api.post('/api/v1/alerts/emotion', payload);
    return data;
  },

  async triggerNoFaceAlert(payload: {
    session_id: string;
    user_name?: string;
    minutes_without_face: number;
    contact_email?: string;
  }) {
    const { data } = await api.post('/api/v1/alerts/no-face', payload);
    return data;
  },
};
