import pandas as pd
from typing import List, Dict, Any
from src.schemas import Question, ModelOutput


# ==========================================
# 1. Базовые метрики для одного вопроса
# ==========================================

def calculate_set_metrics(predicted: List[str], gold: List[str]) -> Dict[str, float]:
    """Рассчитывает Precision, Recall и F1 для двух списков ID (evidence или counterevidence)."""
    pred_set = set(predicted)
    gold_set = set(gold)

    if not gold_set:
        # Если золотых ID нет, а модель что-то вернула — это штраф (False Positive)
        precision = 0.0 if pred_set else 1.0
        recall = 1.0
    elif not pred_set:
        # Модель ничего не вернула, но золото есть — штраф (False Negative)
        precision = 1.0
        recall = 0.0
    else:
        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set)
        recall = tp / len(gold_set)

    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_single_question(question: Question, model_output: ModelOutput) -> Dict[str, Any]:
    """Сравнивает один вывод модели с эталоном и возвращает словарь метрик."""
    status_correct = 1 if question.gold_status == model_output.status else 0

    ev_metrics = calculate_set_metrics(model_output.evidence_ids, question.gold_evidence_ids)
    cev_metrics = calculate_set_metrics(model_output.counterevidence_ids, question.gold_counterevidence_ids)

    # Специальные флаги для сложных статусов
    is_both_correct = 1 if (question.gold_status == "both" and model_output.status == "both") else 0
    is_insuff_correct = 1 if (question.gold_status == "insufficient" and model_output.status == "insufficient") else 0

    return {
        "question_id": question.question_id,
        "status_accuracy": status_correct,
        "evidence_precision": ev_metrics["precision"],
        "evidence_recall": ev_metrics["recall"],
        "evidence_f1": ev_metrics["f1"],
        "counterevidence_recall": cev_metrics["recall"],
        "both_recognized": is_both_correct,
        "insufficient_recognized": is_insuff_correct,
        "confidence": model_output.confidence,
        "predicted_status": model_output.status,
        "gold_status": question.gold_status
    }


# ==========================================
# 2. Агрегация метрик по набору вопросов
# ==========================================

def aggregate_metrics(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Принимает список результатов evaluate_single_question и возвращает агрегированный DataFrame."""
    df = pd.DataFrame(evaluations)

    summary = {
        "Total Questions": len(df),
        "Status Accuracy": df["status_accuracy"].mean(),
        "Evidence Precision": df["evidence_precision"].mean(),
        "Evidence Recall": df["evidence_recall"].mean(),
        "Evidence F1": df["evidence_f1"].mean(),
        "Counterevidence Recall": df["counterevidence_recall"].mean(),
        "Both Recognition Rate": df["both_recognized"].mean() if (df["gold_status"] == "both").any() else 0.0,
        "Insufficient Recognition Rate": df["insufficient_recognized"].mean() if (
                    df["gold_status"] == "insufficient").any() else 0.0,
        "Mean Confidence": df["confidence"].mean()
    }

    return pd.DataFrame([summary])


# ==========================================
# 3. Анализ устойчивости (Стресс-тесты)
# ==========================================

def analyze_stability(baseline_results: List[Dict[str, Any]], stress_results: List[Dict[str, Any]], stress_type: str) -> \
Dict[str, float]:
    """
    Сравнивает baseline и stress-тест.
    stress_type: "permutation", "remove_premise", "add_contradiction"
    """
    df_base = pd.DataFrame(baseline_results).set_index("question_id")
    df_stress = pd.DataFrame(stress_results).set_index("question_id")

    # Доля вопросов, у которых изменился статус
    status_changed = (df_base["predicted_status"] != df_stress["predicted_status"]).mean()

    # Для "remove_premise" ожидаем, что multi-hop вопросы станут insufficient
    # Для "add_contradiction" ожидаем переход к both
    stability_score = 1.0 - status_changed  # Чем меньше изменений, тем стабильнее (кроме целевых тестов)

    return {
        "stress_type": stress_type,
        "status_change_rate": status_changed,
        "stability_score": stability_score,
        "evidence_f1_delta": df_stress["evidence_f1"].mean() - df_base["evidence_f1"].mean()
    }


# ==========================================
# 4. Калибровка уверенности (Confidence Calibration)
# ==========================================

def analyze_confidence_calibration(evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Проверяет, соответствует ли confidence фактической точности.
    Группирует ответы по бакетам уверенности и считает реальную accuracy.
    """
    df = pd.DataFrame(evaluations)

    # Создаем бакеты: [0-0.5], (0.5-0.7], (0.7-0.9], (0.9-1.0]
    bins = [0, 0.5, 0.7, 0.9, 1.0]
    labels = ["Low (0-0.5)", "Medium (0.5-0.7)", "High (0.7-0.9)", "Very High (0.9-1.0)"]
    df["confidence_bucket"] = pd.cut(df["confidence"], bins=bins, labels=labels, include_lowest=True)

    calibration = df.groupby("confidence_bucket", observed=False).agg(
        count=("status_accuracy", "count"),
        actual_accuracy=("status_accuracy", "mean"),
        avg_confidence=("confidence", "mean")
    ).reset_index()

    return calibration


# ==========================================
# 5. Сравнение режимов (A vs B vs C)
# ==========================================

def compare_modes(results_by_mode: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Принимает словарь {mode_name: list_of_evaluations}.
    Возвращает таблицу сравнения режимов.
    """
    comparison_data = []
    for mode, evals in results_by_mode.items():
        agg = aggregate_metrics(evals).iloc[0]
        agg["Mode"] = mode
        comparison_data.append(agg)

    df_comparison = pd.DataFrame(comparison_data)

    # Переупорядочиваем колонки для удобства
    cols = ["Mode", "Status Accuracy", "Evidence F1", "Counterevidence Recall",
            "Both Recognition Rate", "Insufficient Recognition Rate", "Mean Confidence"]

    return df_comparison[cols].round(3)