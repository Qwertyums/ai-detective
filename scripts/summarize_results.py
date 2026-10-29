"""Summarize paired valid runs without conflating errors with insufficient."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_validation import ROOT
from src.experiments import load_records, paired_records
from src.evaluation import aggregate_metrics, analyze_stability, analyze_confidence_calibration


def _summarize(records, paired):
    rows = []
    groups = sorted({(r['study_digest'], r['kind']) for r in paired})
    for study, kind in groups:
        for mode in 'ABC':
            selected = [r for r in paired if r['study_digest'] == study and r['kind'] == kind and r['mode'] == mode]
            if not selected:
                continue
            valid = [r['evaluation'] for r in selected if r['evaluation'] is not None]
            summary = aggregate_metrics(valid)
            row = {'study_digest': study, 'kind': kind, 'mode': mode, 'attempts': len(selected),
                   'model': selected[0]['model'],
                   'provider': selected[0]['settings'].get('provider','local'),
                   'initial_format_failures': sum(bool(r.get('attempts') and ((r['attempts'][0]['error'] and r['attempts'][0]['raw_response'] is not None) or 'quoted_property_names' in r['attempts'][0].get('normalizations',[]))) for r in selected),
                   'normalized_responses': sum(any('quoted_property_names' in a.get('normalizations',[]) for a in r.get('attempts',[])) for r in selected),
                   'repaired_responses': sum(bool(len(r.get('attempts',[])) > 1 and not r['error']) for r in selected),
                   'valid_scored': len(valid), 'errors': sum(r['error'] is not None for r in selected),
                   'unreviewed': sum(not r['reviewed'] for r in selected),
                   'mean_input_tokens': sum(r['usage'].get('input_tokens',0) for r in selected)/len(selected),
                   'mean_response_seconds': sum(r['usage'].get('response_seconds',0) for r in selected)/len(selected)}
            if not summary.empty:
                row.update(summary.iloc[0].to_dict())
                row['accuracy_including_errors'] = sum(x['status_accuracy'] for x in valid)/len(selected)
            rows.append(row)
    calibration = []
    for study, kind in groups:
        if kind != 'baseline':
            continue
        for mode in 'ABC':
            evaluations = [r['evaluation'] for r in paired if r['study_digest'] == study and r['kind'] == kind and r['mode'] == mode and r['evaluation']]
            frame = analyze_confidence_calibration(evaluations)
            if not frame.empty:
                for bucket in frame[frame['count'] > 0].to_dict('records'):
                    bucket['confidence_bucket'] = str(bucket['confidence_bucket'])
                    calibration.append({'study_digest': study, 'mode': mode, **bucket})
    stability = []
    for study, kind in groups:
        if kind == 'baseline':
            continue
        for mode in 'ABC':
            base = [r['evaluation'] for r in paired if r['study_digest'] == study and r['kind'] == 'baseline' and r['mode'] == mode and r['evaluation']]
            changed = [r['evaluation'] for r in paired if r['study_digest'] == study and r['kind'] == kind and r['mode'] == mode and r['evaluation']]
            if not base or not changed:
                continue
            stability.append({'study_digest': study, 'mode': mode, **analyze_stability(base, changed, kind)})
    return {'records': len(records), 'paired_records': len(paired), 'metrics': rows, 'stability': stability, 'calibration': calibration}


def summarize(records):
    # Available per-mode results are useful even before the comparison is complete.
    # Keep the strict A/B/C comparison unchanged for reports and conclusions.
    latest = {}
    for r in sorted(records, key=lambda r: (r['created_at'], r['run_id'])):
        latest[(r['experiment_digest'], r['mode'])] = r
    available = _summarize(records, list(latest.values()))
    result = _summarize(records, paired_records(records))
    coverage = {}
    for r in latest.values():
        row = coverage.setdefault(r['experiment_digest'], {
            'study_digest': r['study_digest'], 'question_id': r['question_id'],
            'case_id': r['case_id'], 'kind': r['kind'], 'modes': []})
        row['modes'].append(r['mode'])
    for row in coverage.values():
        row['modes'] = sorted(row['modes'])
        row['missing_modes'] = sorted(set('ABC') - set(row['modes']))
    result.update(available_records=len(latest), available_metrics=available['metrics'],
                  available_stability=available['stability'], available_calibration=available['calibration'],
                  coverage=list(coverage.values()))
    return result


if __name__ == '__main__':
    result = summarize(load_records())
    path = ROOT/'results/summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
