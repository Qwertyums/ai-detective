"""Three-level ternary paths; time refers to event, not publication."""
from datetime import datetime
START = datetime.fromisoformat('2026-03-14T17:03:00+03:00')
END = datetime.fromisoformat('2026-03-14T19:22:00+03:00')
RULES = {'regulation', 'exception', 'regulation_addendum', 'interpretation'}
STATEMENTS = {'statement', 'correspondence'}

def get_padic_code(claim, documents):
    doc = next(d for d in documents if d.document_id == claim.document_id)
    source = 2 if doc.source_type in RULES or claim.status == 'inferred' else (1 if doc.source_type in STATEMENTS else 0)
    logical = 2 if claim.status == 'disputed' or doc.source_type == 'exception' else (0 if claim.status == 'direct' else 1)
    time = 2 if claim.event_at is None or claim.event_at > END else (0 if claim.event_at < START else 1)
    return source, logical, time

def encode_path(path):
    if len(path) != 3 or any(type(x) is not int or x not in (0, 1, 2) for x in path):
        raise ValueError('A path has three ternary digits')
    return path[0] + 3 * path[1] + 9 * path[2]

def calculate_padic_distance(x, y):
    encode_path(x)
    encode_path(y)
    for level in range(3):
        if x[level] != y[level]:
            return 3 ** -level
    return 0.0

def group_by_padic_branches(claims, documents):
    branches = {}
    for claim in claims:
        key = '[' + ','.join(map(str, get_padic_code(claim, documents))) + ']'
        branches.setdefault(key, []).append(claim)
    return {key: sorted(branches[key], key=lambda c: (c.event_at is None, c.event_at or END, c.claim_id)) for key in sorted(branches)}
