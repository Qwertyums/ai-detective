"""Metrics on valid answers; system failures reported separately."""
import pandas as pd

def calculate_set_metrics(predicted, gold):
    predicted, gold = set(predicted), set(gold)
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else 1.0
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall) if precision + recall else 0.0}

def evaluate_single_question(question, output):
    ev = calculate_set_metrics(output.evidence_ids, question.gold_evidence_ids)
    counter = calculate_set_metrics(output.counterevidence_ids, question.gold_counterevidence_ids)
    return {'question_id': question.question_id, 'status_accuracy': int(output.status == question.gold_status),
            'evidence_precision': ev['precision'], 'evidence_recall': ev['recall'], 'evidence_f1': ev['f1'],
            'counterevidence_recall': counter['recall'] if question.gold_counterevidence_ids else None, 'confidence': output.confidence,
            'gold_status': question.gold_status, 'predicted_status': output.status,
            'evidence_ids': output.evidence_ids, 'counterevidence_ids': output.counterevidence_ids}

def aggregate_metrics(evaluations):
    if not evaluations:
        return pd.DataFrame()
    df = pd.DataFrame(evaluations)
    both = df.gold_status == 'both'
    insuff = df.gold_status == 'insufficient'
    return pd.DataFrame([{
        'Total Questions': len(df), 'Status Accuracy': df.status_accuracy.mean(),
        'Evidence Precision': df.evidence_precision.mean(), 'Evidence Recall': df.evidence_recall.mean(),
        'Evidence F1': df.evidence_f1.mean(), 'Counterevidence Recall': df.counterevidence_recall.mean() if df.counterevidence_recall.notna().any() else None,
        'Both Recognition Rate': (df.loc[both, 'predicted_status'] == 'both').mean() if both.any() else None,
        'Insufficient Recognition Rate': (df.loc[insuff, 'predicted_status'] == 'insufficient').mean() if insuff.any() else None,
        # A threshold is fixed before running experiments; abstention is not a confident assertion.
        'Confident Unsupported Rate': ((df.loc[insuff, 'confidence'] >= .8) & (df.loc[insuff, 'predicted_status'] != 'insufficient')).mean() if insuff.any() else None,
        'Mean Confidence': df.confidence.mean(),
    }])

def analyze_confidence_calibration(evaluations):
    if not evaluations:
        return pd.DataFrame()
    df = pd.DataFrame(evaluations)
    df['confidence_bucket'] = pd.cut(df.confidence, [0, .5, .7, .9, 1], include_lowest=True)
    return df.groupby('confidence_bucket', observed=False).agg(count=('status_accuracy', 'count'),
                actual_accuracy=('status_accuracy', 'mean'), avg_confidence=('confidence', 'mean')).reset_index()

def analyze_stability(baseline_results, stress_results, stress_type):
    base = {x['question_id']: x for x in baseline_results}
    pairs = [(base[x['question_id']], x) for x in stress_results if x['question_id'] in base]
    if not pairs:
        return {'stress_type': stress_type, 'paired_questions': 0}
    change = sum(a['predicted_status'] != b['predicted_status'] for a,b in pairs) / len(pairs)
    jaccard = lambda a,b: len(set(a) & set(b))/len(set(a)|set(b)) if set(a)|set(b) else 1.0
    return {'stress_type': stress_type, 'paired_questions': len(pairs), 'status_change_rate': change,
            'stability_score': 1-change if stress_type == 'permutation' else None,
            'expected_status_accuracy': sum(x['status_accuracy'] for _,x in pairs)/len(pairs),
            'evidence_id_jaccard': sum(jaccard(a['evidence_ids'], b['evidence_ids']) for a,b in pairs)/len(pairs),
            'answer_change_rate': sum(a.get('answer') != b.get('answer') for a,b in pairs)/len(pairs)}

def compare_modes(results_by_mode):
    rows = []
    for mode, results in results_by_mode.items():
        if results:
            rows.append({'Mode': mode, **aggregate_metrics(results).iloc[0].to_dict()})
    return pd.DataFrame(rows)
