from services.emotion_service import EmotionResult, emotion_service


SID = "session_test_precision"


def reset() -> None:
    emotion_service._session_state.pop(SID, None)


def apply_case(raw_emotion: str, confidence: float, scores: dict[str, float], idx: int) -> None:
    result = EmotionResult(
        emotion=raw_emotion,
        confidence=confidence,
        all_scores=scores,
        face_detected=True,
    )
    smoothed = emotion_service._apply_temporal_smoothing(result, SID)
    top = sorted(smoothed.all_scores.items(), key=lambda item: item[1], reverse=True)[:3]
    print(
        f"frame={idx:02d} raw={raw_emotion:8s} conf={confidence:.3f} "
        f"-> chosen={smoothed.emotion:8s} out_conf={smoothed.confidence:.3f} top={top}"
    )


if __name__ == "__main__":
    print("\n[CASE A] anxious baixo (esperado: neutral persistir)\n")
    reset()
    for i in range(1, 7):
        apply_case("anxious", 0.26, {"anxious": 0.28, "neutral": 0.27, "happy": 0.23, "sad": 0.22}, i)

    print("\n[CASE B] happy forte consistente (esperado: migrar para happy)\n")
    reset()
    for i in range(1, 7):
        apply_case("happy", 0.87, {"happy": 0.74, "neutral": 0.16, "surprised": 0.10}, i)

    print("\n[CASE C] oscilacao anxious/sad com gap baixo (esperado: sem troca aleatoria)\n")
    reset()
    sequence = [
        ("neutral", 0.45, {"neutral": 0.41, "anxious": 0.31, "sad": 0.28}),
        ("anxious", 0.30, {"anxious": 0.34, "neutral": 0.33, "sad": 0.33}),
        ("sad", 0.31, {"sad": 0.35, "neutral": 0.33, "anxious": 0.32}),
        ("anxious", 0.29, {"anxious": 0.36, "neutral": 0.34, "sad": 0.30}),
        ("neutral", 0.48, {"neutral": 0.44, "anxious": 0.30, "sad": 0.26}),
        ("neutral", 0.52, {"neutral": 0.50, "anxious": 0.27, "sad": 0.23}),
    ]
    for i, (emotion, conf, score_map) in enumerate(sequence, start=1):
        apply_case(emotion, conf, score_map, i)

    print("\n[CASE D] foto triste estavel a cada segundo (esperado: sad)\n")
    reset()
    sad_sequence = [
        ("sad", 0.47, {"sad": 0.51, "neutral": 0.29, "anxious": 0.20}),
        ("sad", 0.49, {"sad": 0.55, "neutral": 0.27, "anxious": 0.18}),
        ("sad", 0.50, {"sad": 0.56, "neutral": 0.26, "anxious": 0.18}),
        ("sad", 0.48, {"sad": 0.54, "neutral": 0.28, "anxious": 0.18}),
        ("sad", 0.51, {"sad": 0.57, "neutral": 0.25, "anxious": 0.18}),
    ]
    for i, (emotion, conf, score_map) in enumerate(sad_sequence, start=1):
        apply_case(emotion, conf, score_map, i)

    print("\n[CASE E] foto feliz estavel a cada segundo (esperado: happy)\n")
    reset()
    happy_sequence = [
        ("happy", 0.56, {"happy": 0.60, "neutral": 0.26, "surprised": 0.14}),
        ("happy", 0.58, {"happy": 0.63, "neutral": 0.24, "surprised": 0.13}),
        ("happy", 0.59, {"happy": 0.64, "neutral": 0.23, "surprised": 0.13}),
        ("happy", 0.57, {"happy": 0.62, "neutral": 0.25, "surprised": 0.13}),
        ("happy", 0.60, {"happy": 0.65, "neutral": 0.22, "surprised": 0.13}),
    ]
    for i, (emotion, conf, score_map) in enumerate(happy_sequence, start=1):
        apply_case(emotion, conf, score_map, i)

    print("\n[CASE F] tristeza sutil repetida (esperado: sad, nao neutral)\n")
    reset()
    subtle_sad_sequence = [
        ("sad", 0.24, {"sad": 0.33, "neutral": 0.31, "anxious": 0.18, "happy": 0.18}),
        ("sad", 0.25, {"sad": 0.35, "neutral": 0.29, "anxious": 0.18, "happy": 0.18}),
        ("sad", 0.26, {"sad": 0.36, "neutral": 0.28, "anxious": 0.18, "happy": 0.18}),
        ("sad", 0.25, {"sad": 0.34, "neutral": 0.29, "anxious": 0.18, "happy": 0.19}),
        ("sad", 0.27, {"sad": 0.37, "neutral": 0.27, "anxious": 0.18, "happy": 0.18}),
    ]
    for i, (emotion, conf, score_map) in enumerate(subtle_sad_sequence, start=1):
        apply_case(emotion, conf, score_map, i)

    print("\n[CASE G] felicidade sutil repetida (esperado: happy, nao neutral)\n")
    reset()
    subtle_happy_sequence = [
        ("happy", 0.23, {"happy": 0.34, "neutral": 0.30, "surprised": 0.18, "sad": 0.18}),
        ("happy", 0.24, {"happy": 0.36, "neutral": 0.28, "surprised": 0.18, "sad": 0.18}),
        ("happy", 0.25, {"happy": 0.37, "neutral": 0.27, "surprised": 0.18, "sad": 0.18}),
        ("happy", 0.24, {"happy": 0.35, "neutral": 0.29, "surprised": 0.18, "sad": 0.18}),
        ("happy", 0.26, {"happy": 0.38, "neutral": 0.26, "surprised": 0.18, "sad": 0.18}),
    ]
    for i, (emotion, conf, score_map) in enumerate(subtle_happy_sequence, start=1):
        apply_case(emotion, conf, score_map, i)
