from typing import List, Dict, Tuple
from src.schemas import Claim


def get_padic_code(claim: Claim) -> Tuple[int, int, int]:
    """Определяет p-адический код [a0, a1, a2] для утверждения."""
    # a0: тип источника (0: системный/док, 1: свидетельство, 2: правило)
    if claim.document_id in ["D05", "D09", "D10"]:  # логи, отчеты СБ, отчеты
        a0 = 0
    elif claim.document_id in ["D03"]:  # свидетельства
        a0 = 1
    else:  # регламенты, описания, переписка
        a0 = 2 if "regulation" in claim.document_id.lower() or "exception" in claim.document_id.lower() else 0

    # a1: логический статус (0: установленный факт/direct, 1: сообщение/reported/inferred, 2: спорное/disputed)
    if claim.status == "direct":
        a1 = 0
    elif claim.status in ["reported", "inferred"]:
        a1 = 1
    else:
        a1 = 2

    # a2: время (0: до события, 1: во время, 2: после/неизвестно)
    # Упрощенная эвристика по времени создания документа относительно инцидента (14 марта 17:00)
    if "2026-01" in claim.document_id or "2026-02" in claim.document_id or "16:" in claim.document_id:  # до
        a2 = 0
    elif "17:" in claim.document_id or "18:" in claim.document_id:  # во время
        a2 = 1
    else:  # после (19:, 20:, 15 марта)
        a2 = 2

    return (a0, a1, a2)


def calculate_padic_distance(code1: Tuple[int, int, int], code2: Tuple[int, int, int]) -> float:
    """Вычисляет d_3(x,y) = 3^(-k), где k - число совпадающих начальных уровней."""
    if code1 == code2:
        return 0.0
    k = 0
    for i in range(3):
        if code1[i] == code2[i]:
            k += 1
        else:
            break
    return 3 ** (-k)


def group_by_padic_branches(claims: List[Claim]) -> Dict[str, List[Claim]]:
    """Группирует утверждения по ветвям [a0, a1, a2] и сортирует внутри по времени."""
    branches = {}
    for claim in claims:
        code = get_padic_code(claim)
        branch_key = f"[{code[0]},{code[1]},{code[2]}]"
        if branch_key not in branches:
            branches[branch_key] = []
        branches[branch_key].append(claim)

    for branch in branches.values():
        branch.sort(key=lambda x: x.created_at if hasattr(x, 'created_at') else x.claim_id)
    return branches