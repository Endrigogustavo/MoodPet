import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated, Easing } from 'react-native';
import Svg, {
  Circle, Ellipse, Path, G, Rect,
} from 'react-native-svg';
import { EMOTION_COLORS } from '../theme';

interface VirtualPetProps {
  emotion: string;
  petType?: 'dog' | 'cat' | 'bunny' | 'bear' | 'fox' | 'panda' | 'owl' | 'seal';
  size?: number;
  animate?: boolean;
}

// ── Animation values per emotion ─────────────────────────────────────────────
const EMOTION_ANIMATIONS: Record<string, {
  bounce: number;
  shake: boolean;
  pulse: boolean;
  float: boolean;
  tilt: boolean;
  aura: boolean;
}> = {
  happy:     { bounce: 1, shake: false, pulse: false, float: true,  tilt: true,  aura: true },
  sad:       { bounce: 0, shake: false, pulse: true,  float: false, tilt: false, aura: true },
  angry:     { bounce: 0, shake: true,  pulse: false, float: false, tilt: false, aura: false },
  anxious:   { bounce: 0, shake: false, pulse: true,  float: false, tilt: true,  aura: true },
  neutral:   { bounce: 0, shake: false, pulse: false, float: true,  tilt: false, aura: false },
  surprised: { bounce: 1, shake: false, pulse: false, float: false, tilt: true,  aura: true },
  disgusted: { bounce: 0, shake: false, pulse: false, float: false, tilt: false, aura: false },
  fearful:   { bounce: 0, shake: true,  pulse: true,  float: false, tilt: true,  aura: true },
};

// ── Pet face renderer ─────────────────────────────────────────────────────────
function DogFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.38;
  const color = EMOTION_COLORS[emotion]?.bg || EMOTION_COLORS.neutral.bg;

  // Eyes
  const eyeY = cy - r * 0.15;
  const eyeXL = cx - r * 0.35;
  const eyeXR = cx + r * 0.35;

  // Mouth shapes per emotion
  const getMouth = () => {
    const mx = cx;
    const my = cy + r * 0.3;
    const mw = r * 0.5;
    switch (emotion) {
      case 'happy':
        return <Path d={`M${mx - mw} ${my} Q${mx} ${my + mw * 0.8} ${mx + mw} ${my}`}
                     stroke="#5D4037" strokeWidth={s * 0.025} fill="none" strokeLinecap="round" />;
      case 'sad':
        return <Path d={`M${mx - mw} ${my + mw * 0.4} Q${mx} ${my - mw * 0.4} ${mx + mw} ${my + mw * 0.4}`}
                     stroke="#5D4037" strokeWidth={s * 0.025} fill="none" strokeLinecap="round" />;
      case 'angry':
        return <Path d={`M${mx - mw * 0.8} ${my} L${mx + mw * 0.8} ${my}`}
                     stroke="#5D4037" strokeWidth={s * 0.025} fill="none" strokeLinecap="round" />;
      case 'surprised':
        return <Ellipse cx={mx} cy={my + r * 0.05} rx={mw * 0.4} ry={mw * 0.5}
                        fill="#5D4037" opacity={0.7} />;
      case 'anxious':
        return <Path d={`M${mx - mw} ${my} Q${mx - mw * 0.3} ${my + mw * 0.3} ${mx} ${my} Q${mx + mw * 0.3} ${my - mw * 0.3} ${mx + mw} ${my}`}
                     stroke="#5D4037" strokeWidth={s * 0.025} fill="none" strokeLinecap="round" />;
      default:
        return <Path d={`M${mx - mw * 0.6} ${my} Q${mx} ${my + mw * 0.2} ${mx + mw * 0.6} ${my}`}
                     stroke="#5D4037" strokeWidth={s * 0.022} fill="none" strokeLinecap="round" />;
    }
  };

  const getEyes = () => {
    if (emotion === 'happy') {
      // Happy eyes = curved
      return (
        <G>
          <Path d={`M${eyeXL - r * 0.12} ${eyeY} Q${eyeXL} ${eyeY - r * 0.15} ${eyeXL + r * 0.12} ${eyeY}`}
                stroke="#5D4037" strokeWidth={s * 0.03} fill="none" strokeLinecap="round" />
          <Path d={`M${eyeXR - r * 0.12} ${eyeY} Q${eyeXR} ${eyeY - r * 0.15} ${eyeXR + r * 0.12} ${eyeY}`}
                stroke="#5D4037" strokeWidth={s * 0.03} fill="none" strokeLinecap="round" />
        </G>
      );
    }
    if (emotion === 'sad') {
      return (
        <G>
          <Circle cx={eyeXL} cy={eyeY} r={r * 0.12} fill="#5D4037" />
          <Circle cx={eyeXR} cy={eyeY} r={r * 0.12} fill="#5D4037" />
          {/* Tear drops */}
          <Ellipse cx={eyeXL + r * 0.04} cy={eyeY + r * 0.2} rx={r * 0.05} ry={r * 0.12}
                   fill="#74B9FF" opacity={0.8} />
        </G>
      );
    }
    if (emotion === 'surprised') {
      return (
        <G>
          <Circle cx={eyeXL} cy={eyeY} r={r * 0.17} fill="white" />
          <Circle cx={eyeXL} cy={eyeY} r={r * 0.1} fill="#5D4037" />
          <Circle cx={eyeXR} cy={eyeY} r={r * 0.17} fill="white" />
          <Circle cx={eyeXR} cy={eyeY} r={r * 0.1} fill="#5D4037" />
        </G>
      );
    }
    return (
      <G>
        <Circle cx={eyeXL} cy={eyeY} r={r * 0.13} fill="#5D4037" />
        <Circle cx={eyeXL - r * 0.04} cy={eyeY - r * 0.04} r={r * 0.04} fill="white" />
        <Circle cx={eyeXR} cy={eyeY} r={r * 0.13} fill="#5D4037" />
        <Circle cx={eyeXR - r * 0.04} cy={eyeY - r * 0.04} r={r * 0.04} fill="white" />
      </G>
    );
  };

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      {/* Ears */}
      <Ellipse cx={cx - r * 0.65} cy={cy - r * 0.75} rx={r * 0.28} ry={r * 0.38}
               fill={color} opacity={0.85} />
      <Ellipse cx={cx + r * 0.65} cy={cy - r * 0.75} rx={r * 0.28} ry={r * 0.38}
               fill={color} opacity={0.85} />

      {/* Head */}
      <Circle cx={cx} cy={cy} r={r} fill={color} />

      {/* Inner ears */}
      <Ellipse cx={cx - r * 0.65} cy={cy - r * 0.75} rx={r * 0.16} ry={r * 0.24}
               fill="#FFCCBC" opacity={0.7} />
      <Ellipse cx={cx + r * 0.65} cy={cy - r * 0.75} rx={r * 0.16} ry={r * 0.24}
               fill="#FFCCBC" opacity={0.7} />

      {/* Snout */}
      <Ellipse cx={cx} cy={cy + r * 0.18} rx={r * 0.35} ry={r * 0.25}
               fill="#FFCCBC" opacity={0.8} />

      {/* Nose */}
      <Ellipse cx={cx} cy={cy + r * 0.05} rx={r * 0.12} ry={r * 0.08}
               fill="#5D4037" />
      <Circle cx={cx - r * 0.04} cy={cy + r * 0.02} r={r * 0.03} fill="white" opacity={0.6} />

      {/* Eyes */}
      {getEyes()}

      {/* Blush */}
      {emotion === 'happy' && (
        <G>
          <Ellipse cx={cx - r * 0.55} cy={cy + r * 0.12} rx={r * 0.18} ry={r * 0.1}
                   fill="#FFB3BA" opacity={0.5} />
          <Ellipse cx={cx + r * 0.55} cy={cy + r * 0.12} rx={r * 0.18} ry={r * 0.1}
                   fill="#FFB3BA" opacity={0.5} />
        </G>
      )}

      {/* Mouth */}
      {getMouth()}
    </Svg>
  );
}

function CatFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.38;
  const color = EMOTION_COLORS[emotion]?.bg || EMOTION_COLORS.neutral.bg;

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      {/* Pointy ears */}
      <Path d={`M${cx - r * 0.7} ${cy - r * 0.6} L${cx - r * 0.35} ${cy - r * 1.1} L${cx - r * 0.05} ${cy - r * 0.7}`}
            fill={color} />
      <Path d={`M${cx + r * 0.7} ${cy - r * 0.6} L${cx + r * 0.35} ${cy - r * 1.1} L${cx + r * 0.05} ${cy - r * 0.7}`}
            fill={color} />
      {/* Inner ear */}
      <Path d={`M${cx - r * 0.6} ${cy - r * 0.65} L${cx - r * 0.35} ${cy - r * 1.0} L${cx - r * 0.1} ${cy - r * 0.72}`}
            fill="#FFCCBC" opacity={0.6} />
      <Path d={`M${cx + r * 0.6} ${cy - r * 0.65} L${cx + r * 0.35} ${cy - r * 1.0} L${cx + r * 0.1} ${cy - r * 0.72}`}
            fill="#FFCCBC" opacity={0.6} />

      {/* Head */}
      <Circle cx={cx} cy={cy} r={r} fill={color} />

      {/* Cat eyes (almond shaped) */}
      <Ellipse cx={cx - r * 0.35} cy={cy - r * 0.15} rx={r * 0.18} ry={r * 0.12}
               fill="#5D4037" />
      <Ellipse cx={cx + r * 0.35} cy={cy - r * 0.15} rx={r * 0.18} ry={r * 0.12}
               fill="#5D4037" />
      <Circle cx={cx - r * 0.3} cy={cy - r * 0.18} r={r * 0.04} fill="white" />
      <Circle cx={cx + r * 0.3} cy={cy - r * 0.18} r={r * 0.04} fill="white" />

      {/* Tiny nose */}
      <Path d={`M${cx} ${cy + r * 0.08} L${cx - r * 0.08} ${cy + r * 0.18} L${cx + r * 0.08} ${cy + r * 0.18} Z`}
            fill="#FF8A80" />

      {/* Whiskers */}
      <Path d={`M${cx - r * 0.15} ${cy + r * 0.15} L${cx - r * 0.7} ${cy + r * 0.1}`}
            stroke="#9E9E9E" strokeWidth={1.5} opacity={0.6} />
      <Path d={`M${cx - r * 0.15} ${cy + r * 0.22} L${cx - r * 0.7} ${cy + r * 0.25}`}
            stroke="#9E9E9E" strokeWidth={1.5} opacity={0.6} />
      <Path d={`M${cx + r * 0.15} ${cy + r * 0.15} L${cx + r * 0.7} ${cy + r * 0.1}`}
            stroke="#9E9E9E" strokeWidth={1.5} opacity={0.6} />
      <Path d={`M${cx + r * 0.15} ${cy + r * 0.22} L${cx + r * 0.7} ${cy + r * 0.25}`}
            stroke="#9E9E9E" strokeWidth={1.5} opacity={0.6} />

      {/* Mouth */}
      {emotion === 'happy'
        ? <Path d={`M${cx - r * 0.3} ${cy + r * 0.32} Q${cx} ${cy + r * 0.5} ${cx + r * 0.3} ${cy + r * 0.32}`}
                stroke="#5D4037" strokeWidth={s * 0.025} fill="none" strokeLinecap="round" />
        : <Path d={`M${cx - r * 0.15} ${cy + r * 0.35} L${cx} ${cy + r * 0.32} L${cx + r * 0.15} ${cy + r * 0.35}`}
                stroke="#5D4037" strokeWidth={s * 0.022} fill="none" strokeLinecap="round" />
      }

      {/* Blush */}
      {(emotion === 'happy' || emotion === 'surprised') && (
        <G>
          <Ellipse cx={cx - r * 0.58} cy={cy + r * 0.12} rx={r * 0.15} ry={r * 0.09}
                   fill="#FFB3BA" opacity={0.5} />
          <Ellipse cx={cx + r * 0.58} cy={cy + r * 0.12} rx={r * 0.15} ry={r * 0.09}
                   fill="#FFB3BA" opacity={0.5} />
        </G>
      )}
    </Svg>
  );
}

function BunnyFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.36;
  const color = '#E6D8FF';

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <Ellipse cx={cx - r * 0.32} cy={cy - r * 1.05} rx={r * 0.18} ry={r * 0.52} fill={color} />
      <Ellipse cx={cx + r * 0.32} cy={cy - r * 1.05} rx={r * 0.18} ry={r * 0.52} fill={color} />
      <Ellipse cx={cx - r * 0.32} cy={cy - r * 1.05} rx={r * 0.09} ry={r * 0.36} fill="#FFB6C1" opacity={0.7} />
      <Ellipse cx={cx + r * 0.32} cy={cy - r * 1.05} rx={r * 0.09} ry={r * 0.36} fill="#FFB6C1" opacity={0.7} />

      <Circle cx={cx} cy={cy} r={r} fill={color} />
      <Circle cx={cx - r * 0.32} cy={cy - r * 0.12} r={r * 0.11} fill="#3E2C41" />
      <Circle cx={cx + r * 0.32} cy={cy - r * 0.12} r={r * 0.11} fill="#3E2C41" />

      <Ellipse cx={cx} cy={cy + r * 0.24} rx={r * 0.24} ry={r * 0.18} fill="#FFF5FF" />
      <Circle cx={cx} cy={cy + r * 0.2} r={r * 0.06} fill="#FF8AAE" />
      <Path
        d={`M${cx - r * 0.18} ${cy + r * 0.34} Q${cx} ${cy + r * 0.45} ${cx + r * 0.18} ${cy + r * 0.34}`}
        stroke="#5D4A66"
        strokeWidth={s * 0.02}
        fill="none"
        strokeLinecap="round"
      />

      <Circle cx={cx - r * 0.58} cy={cy + r * 0.04} r={r * 0.08} fill="#FFB6C1" opacity={0.35} />
      <Circle cx={cx + r * 0.58} cy={cy + r * 0.04} r={r * 0.08} fill="#FFB6C1" opacity={0.35} />
    </Svg>
  );
}

function BearFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.38;
  const color = '#B9855B';

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <Circle cx={cx - r * 0.74} cy={cy - r * 0.62} r={r * 0.26} fill={color} />
      <Circle cx={cx + r * 0.74} cy={cy - r * 0.62} r={r * 0.26} fill={color} />
      <Circle cx={cx - r * 0.74} cy={cy - r * 0.62} r={r * 0.12} fill="#EACAA8" opacity={0.9} />
      <Circle cx={cx + r * 0.74} cy={cy - r * 0.62} r={r * 0.12} fill="#EACAA8" opacity={0.9} />

      <Circle cx={cx} cy={cy} r={r} fill={color} />
      <Ellipse cx={cx} cy={cy + r * 0.22} rx={r * 0.32} ry={r * 0.24} fill="#EACAA8" />

      <Circle cx={cx - r * 0.3} cy={cy - r * 0.1} r={r * 0.1} fill="#3E2723" />
      <Circle cx={cx + r * 0.3} cy={cy - r * 0.1} r={r * 0.1} fill="#3E2723" />
      <Ellipse cx={cx} cy={cy + r * 0.15} rx={r * 0.1} ry={r * 0.07} fill="#3E2723" />
      <Path
        d={`M${cx - r * 0.18} ${cy + r * 0.3} Q${cx} ${cy + r * 0.42} ${cx + r * 0.18} ${cy + r * 0.3}`}
        stroke="#3E2723"
        strokeWidth={s * 0.02}
        fill="none"
        strokeLinecap="round"
      />

      <Circle cx={cx - r * 0.52} cy={cy + r * 0.04} r={r * 0.07} fill="#FFAB91" opacity={0.28} />
      <Circle cx={cx + r * 0.52} cy={cy + r * 0.04} r={r * 0.07} fill="#FFAB91" opacity={0.28} />
    </Svg>
  );
}

function FoxFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.38;
  const color = '#FF8C42';

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <Path d={`M${cx - r * 0.75} ${cy - r * 0.45} L${cx - r * 0.35} ${cy - r * 1.0} L${cx - r * 0.05} ${cy - r * 0.35} Z`} fill={color} />
      <Path d={`M${cx + r * 0.75} ${cy - r * 0.45} L${cx + r * 0.35} ${cy - r * 1.0} L${cx + r * 0.05} ${cy - r * 0.35} Z`} fill={color} />
      <Path d={`M${cx - r * 0.64} ${cy - r * 0.46} L${cx - r * 0.35} ${cy - r * 0.88} L${cx - r * 0.14} ${cy - r * 0.44} Z`} fill="#FFF3E0" />
      <Path d={`M${cx + r * 0.64} ${cy - r * 0.46} L${cx + r * 0.35} ${cy - r * 0.88} L${cx + r * 0.14} ${cy - r * 0.44} Z`} fill="#FFF3E0" />

      <Circle cx={cx} cy={cy} r={r} fill={color} />
      <Path d={`M${cx - r * 0.72} ${cy - r * 0.05} Q${cx} ${cy + r * 0.55} ${cx + r * 0.72} ${cy - r * 0.05} Q${cx} ${cy + r * 0.12} ${cx - r * 0.72} ${cy - r * 0.05} Z`} fill="#FFF3E0" />

      <Ellipse cx={cx - r * 0.3} cy={cy - r * 0.12} rx={r * 0.12} ry={r * 0.1} fill="#2F2F2F" />
      <Ellipse cx={cx + r * 0.3} cy={cy - r * 0.12} rx={r * 0.12} ry={r * 0.1} fill="#2F2F2F" />
      <Path d={`M${cx} ${cy + r * 0.08} L${cx - r * 0.08} ${cy + r * 0.2} L${cx + r * 0.08} ${cy + r * 0.2} Z`} fill="#2F2F2F" />
    </Svg>
  );
}

function PandaFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.38;

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <Circle cx={cx - r * 0.7} cy={cy - r * 0.62} r={r * 0.24} fill="#1E1E1E" />
      <Circle cx={cx + r * 0.7} cy={cy - r * 0.62} r={r * 0.24} fill="#1E1E1E" />

      <Circle cx={cx} cy={cy} r={r} fill="#F5F5F5" />
      <Ellipse cx={cx - r * 0.32} cy={cy - r * 0.1} rx={r * 0.2} ry={r * 0.16} fill="#1E1E1E" />
      <Ellipse cx={cx + r * 0.32} cy={cy - r * 0.1} rx={r * 0.2} ry={r * 0.16} fill="#1E1E1E" />
      <Circle cx={cx - r * 0.32} cy={cy - r * 0.1} r={r * 0.07} fill="#FFF" />
      <Circle cx={cx + r * 0.32} cy={cy - r * 0.1} r={r * 0.07} fill="#FFF" />
      <Ellipse cx={cx} cy={cy + r * 0.15} rx={r * 0.1} ry={r * 0.07} fill="#1E1E1E" />
      <Path
        d={`M${cx - r * 0.16} ${cy + r * 0.3} Q${cx} ${cy + r * 0.42} ${cx + r * 0.16} ${cy + r * 0.3}`}
        stroke="#1E1E1E"
        strokeWidth={s * 0.02}
        fill="none"
        strokeLinecap="round"
      />
    </Svg>
  );
}

function OwlFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.36;
  const color = '#8E6D5A';

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <Path d={`M${cx - r * 0.6} ${cy - r * 0.85} L${cx - r * 0.35} ${cy - r * 1.15} L${cx - r * 0.1} ${cy - r * 0.82} Z`} fill={color} />
      <Path d={`M${cx + r * 0.6} ${cy - r * 0.85} L${cx + r * 0.35} ${cy - r * 1.15} L${cx + r * 0.1} ${cy - r * 0.82} Z`} fill={color} />

      <Ellipse cx={cx} cy={cy} rx={r * 0.95} ry={r} fill={color} />

      <Circle cx={cx - r * 0.28} cy={cy - r * 0.12} r={r * 0.22} fill="#FFF8E1" />
      <Circle cx={cx + r * 0.28} cy={cy - r * 0.12} r={r * 0.22} fill="#FFF8E1" />
      <Circle cx={cx - r * 0.28} cy={cy - r * 0.12} r={r * 0.09} fill="#3E2723" />
      <Circle cx={cx + r * 0.28} cy={cy - r * 0.12} r={r * 0.09} fill="#3E2723" />

      <Path d={`M${cx} ${cy + r * 0.02} L${cx - r * 0.09} ${cy + r * 0.18} L${cx + r * 0.09} ${cy + r * 0.18} Z`} fill="#F4B400" />
      <Path
        d={`M${cx - r * 0.18} ${cy + r * 0.3} Q${cx} ${cy + r * 0.4} ${cx + r * 0.18} ${cy + r * 0.3}`}
        stroke="#FFF8E1"
        strokeWidth={s * 0.017}
        fill="none"
        strokeLinecap="round"
      />
    </Svg>
  );
}

function SealFace({ emotion, size }: { emotion: string; size: number }) {
  const s = size;
  const cx = s / 2;
  const cy = s / 2;
  const r = s * 0.38;
  const bodyColor = '#9CB4C5';
  const bodyShadow = '#839AAC';
  const muzzle = '#E8F0F5';

  return (
    <Svg width={s} height={s} viewBox={`0 0 ${s} ${s}`}>
      <Ellipse cx={cx - r * 0.86} cy={cy + r * 0.2} rx={r * 0.2} ry={r * 0.11} fill={bodyShadow} opacity={0.9} />
      <Ellipse cx={cx + r * 0.86} cy={cy + r * 0.2} rx={r * 0.2} ry={r * 0.11} fill={bodyShadow} opacity={0.9} />

      <Ellipse cx={cx} cy={cy} rx={r * 1.02} ry={r * 0.88} fill={bodyColor} />
      <Ellipse cx={cx} cy={cy + r * 0.36} rx={r * 0.88} ry={r * 0.38} fill={bodyShadow} opacity={0.22} />
      <Ellipse cx={cx} cy={cy + r * 0.18} rx={r * 0.48} ry={r * 0.34} fill={muzzle} />
      <Ellipse cx={cx - r * 0.58} cy={cy - r * 0.05} rx={r * 0.12} ry={r * 0.08} fill="#FFC6CF" opacity={0.4} />
      <Ellipse cx={cx + r * 0.58} cy={cy - r * 0.05} rx={r * 0.12} ry={r * 0.08} fill="#FFC6CF" opacity={0.4} />

      <Circle cx={cx - r * 0.28} cy={cy - r * 0.12} r={r * 0.1} fill="#2A3440" />
      <Circle cx={cx + r * 0.28} cy={cy - r * 0.12} r={r * 0.1} fill="#2A3440" />
      <Circle cx={cx - r * 0.31} cy={cy - r * 0.15} r={r * 0.03} fill="#FFF" opacity={0.85} />
      <Circle cx={cx + r * 0.25} cy={cy - r * 0.15} r={r * 0.03} fill="#FFF" opacity={0.85} />

      <Ellipse cx={cx} cy={cy + r * 0.1} rx={r * 0.1} ry={r * 0.08} fill="#2A3440" />
      {emotion === 'happy' ? (
        <Path
          d={`M${cx - r * 0.24} ${cy + r * 0.28} Q${cx} ${cy + r * 0.45} ${cx + r * 0.24} ${cy + r * 0.28}`}
          stroke="#2A3440"
          strokeWidth={s * 0.02}
          fill="none"
          strokeLinecap="round"
        />
      ) : (
        <Path
          d={`M${cx - r * 0.2} ${cy + r * 0.31} Q${cx} ${cy + r * 0.25} ${cx + r * 0.2} ${cy + r * 0.31}`}
          stroke="#2A3440"
          strokeWidth={s * 0.018}
          fill="none"
          strokeLinecap="round"
        />
      )}

      <Path d={`M${cx - r * 0.18} ${cy + r * 0.17} L${cx - r * 0.58} ${cy + r * 0.13}`} stroke="#6B7D8A" strokeWidth={1.6} opacity={0.75} />
      <Path d={`M${cx - r * 0.18} ${cy + r * 0.23} L${cx - r * 0.58} ${cy + r * 0.27}`} stroke="#6B7D8A" strokeWidth={1.6} opacity={0.75} />
      <Path d={`M${cx + r * 0.18} ${cy + r * 0.17} L${cx + r * 0.58} ${cy + r * 0.13}`} stroke="#6B7D8A" strokeWidth={1.6} opacity={0.75} />
      <Path d={`M${cx + r * 0.18} ${cy + r * 0.23} L${cx + r * 0.58} ${cy + r * 0.27}`} stroke="#6B7D8A" strokeWidth={1.6} opacity={0.75} />

      <Ellipse cx={cx - r * 0.42} cy={cy - r * 0.58} rx={r * 0.08} ry={r * 0.05} fill="#FFFFFF" opacity={0.2} />
    </Svg>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export const VirtualPet: React.FC<VirtualPetProps> = ({
  emotion,
  petType = 'dog',
  size = 200,
  animate = true,
}) => {
  const bounceAnim = useRef(new Animated.Value(0)).current;
  const shakeAnim = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const floatAnim = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const auraAnim = useRef(new Animated.Value(0.2)).current;
  const tiltAnim = useRef(new Animated.Value(0)).current;

  const config = EMOTION_ANIMATIONS[emotion] || EMOTION_ANIMATIONS.neutral;
  const emotionColor = EMOTION_COLORS[emotion]?.bg || EMOTION_COLORS.neutral.bg;
  const mountedRef = useRef(false);

  useEffect(() => {
    if (!animate) return;

    if (!mountedRef.current) {
      mountedRef.current = true;
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 7,
        tension: 90,
        useNativeDriver: true,
      }).start();
    } else {
      scaleAnim.setValue(1);
    }

    // Bounce (happy/surprised)
    if (config.bounce) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(bounceAnim, { toValue: -16, duration: 400, easing: Easing.out(Easing.quad), useNativeDriver: true }),
          Animated.timing(bounceAnim, { toValue: 0, duration: 400, easing: Easing.in(Easing.quad), useNativeDriver: true }),
        ])
      ).start();
    } else {
      bounceAnim.setValue(0);
    }

    // Shake (angry/fearful)
    if (config.shake) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(shakeAnim, { toValue: 5, duration: 80, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue: -5, duration: 80, useNativeDriver: true }),
          Animated.timing(shakeAnim, { toValue: 0, duration: 80, useNativeDriver: true }),
          Animated.delay(400),
        ])
      ).start();
    } else {
      shakeAnim.setValue(0);
    }

    // Pulse (sad/anxious)
    if (config.pulse) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.06, duration: 1200, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 0.94, duration: 1200, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        ])
      ).start();
    } else {
      pulseAnim.setValue(1);
    }

    // Float (happy/neutral)
    if (config.float) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(floatAnim, { toValue: -8, duration: 1800, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
          Animated.timing(floatAnim, { toValue: 0, duration: 1800, easing: Easing.inOut(Easing.sin), useNativeDriver: true }),
        ])
      ).start();
    } else {
      floatAnim.setValue(0);
    }

    if (config.tilt) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(tiltAnim, { toValue: 1, duration: 800, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
          Animated.timing(tiltAnim, { toValue: -1, duration: 800, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
          Animated.timing(tiltAnim, { toValue: 0, duration: 800, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        ])
      ).start();
    } else {
      tiltAnim.setValue(0);
    }

    if (config.aura) {
      Animated.loop(
        Animated.sequence([
          Animated.timing(auraAnim, { toValue: 0.35, duration: 1200, useNativeDriver: true }),
          Animated.timing(auraAnim, { toValue: 0.15, duration: 1200, useNativeDriver: true }),
        ])
      ).start();
    } else {
      auraAnim.setValue(0.08);
    }

    return () => {
      bounceAnim.stopAnimation();
      shakeAnim.stopAnimation();
      pulseAnim.stopAnimation();
      floatAnim.stopAnimation();
      tiltAnim.stopAnimation();
      auraAnim.stopAnimation();
    };
  }, [emotion, animate]);

  const renderPet = () => {
    switch (petType) {
      case 'cat': return <CatFace emotion={emotion} size={size} />;
      case 'bunny': return <BunnyFace emotion={emotion} size={size} />;
      case 'bear': return <BearFace emotion={emotion} size={size} />;
      case 'fox': return <FoxFace emotion={emotion} size={size} />;
      case 'panda': return <PandaFace emotion={emotion} size={size} />;
      case 'owl': return <OwlFace emotion={emotion} size={size} />;
      case 'seal': return <SealFace emotion={emotion} size={size} />;
      default: return <DogFace emotion={emotion} size={size} />;
    }
  };

  return (
    <Animated.View
      style={[
        styles.container,
        {
          width: size,
          height: size,
          transform: [
            { translateY: Animated.add(bounceAnim, floatAnim) },
            { translateX: shakeAnim },
            {
              rotate: tiltAnim.interpolate({
                inputRange: [-1, 0, 1],
                outputRange: ['-4deg', '0deg', '4deg'],
              }),
            },
            { scale: Animated.multiply(scaleAnim, pulseAnim) },
          ],
        },
      ]}
    >
      <View
        style={[
          styles.floorShadow,
          {
            width: size * 0.52,
            height: size * 0.12,
            borderRadius: size * 0.07,
          },
        ]}
      />

      <Animated.View
        style={[
          styles.aura,
          {
            width: size * 0.86,
            height: size * 0.86,
            borderRadius: size * 0.43,
            backgroundColor: emotionColor,
            opacity: auraAnim,
            transform: [
              {
                scale: auraAnim.interpolate({
                  inputRange: [0.1, 0.35],
                  outputRange: [0.9, 1.06],
                }),
              },
            ],
          },
        ]}
      />

      <Animated.View
        style={[
          styles.sparkle,
          styles.sparkleOne,
          {
            opacity: 0.28,
            transform: [
              {
                translateY: floatAnim.interpolate({
                  inputRange: [-8, 0],
                  outputRange: [-2, 2],
                }),
              },
            ],
          },
        ]}
      />
      <Animated.View
        style={[
          styles.sparkle,
          styles.sparkleTwo,
          {
            opacity: 0.22,
            transform: [
              {
                translateY: floatAnim.interpolate({
                  inputRange: [-8, 0],
                  outputRange: [2, -2],
                }),
              },
            ],
          },
        ]}
      />
      <Animated.View
        style={[
          styles.sparkle,
          styles.sparkleThree,
          {
            opacity: 0.2,
          },
        ]}
      />
      {renderPet()}
    </Animated.View>
  );
};

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  aura: {
    position: 'absolute',
  },
  floorShadow: {
    position: 'absolute',
    bottom: 4,
    backgroundColor: '#0C22311A',
  },
  sparkle: {
    position: 'absolute',
    width: 8,
    height: 8,
    borderRadius: 5,
    backgroundColor: '#FFFFFF',
  },
  sparkleOne: {
    top: 26,
    left: 18,
  },
  sparkleTwo: {
    top: 42,
    right: 16,
    width: 6,
    height: 6,
    borderRadius: 4,
  },
  sparkleThree: {
    top: 18,
    right: 44,
    width: 5,
    height: 5,
    borderRadius: 3,
  },
});
