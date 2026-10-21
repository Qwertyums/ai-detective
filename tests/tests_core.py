import pytest
from src.padic import calculate_padic_distance, get_padic_code
from src.schemas import ModelOutput, ValidationError, Claim
from src.retrieval import bm25_retrieve

def test_padic_distance_levels():
    assert calculate_padic_distance((0, 1, 2), (0, 1, 2)) == 0.0
    assert calculate_padic_distance((0, 1, 2), (1, 1, 2)) == 1.0      # k=0
    assert calculate_padic_distance((0, 1, 2), (0, 2, 2)) == pytest.approx(1/3) # k=1
    assert calculate_padic_distance((0, 1, 2), (0, 1, 0)) == pytest.approx(1/9) # k=2

def test_ultrametric_inequality():
    x, y, z = (0, 1, 2), (0, 2, 2), (1, 2, 2)
    d_xy = calculate_padic_distance(x, y) # 1/3
    d_yz = calculate_padic_distance(y, z) # 1
    d_xz = calculate_padic_distance(x, z) # 1
    assert d_xz <= max(d_xy, d_yz)

def test_schema_both_requires_counterevidence():
    with pytest.raises(ValidationError):
        ModelOutput(
            question_id="Q1", status="both", answer="test",
            evidence_ids=["C1"], counterevidence_ids=[], # Ошибка: пусто
            confidence=0.9, explanation="test"
        )

def test_schema_confidence_range():
    with pytest.raises(ValidationError):
        ModelOutput(
            question_id="Q1", status="entailed", answer="test",
            evidence_ids=["C1"], counterevidence_ids=[],
            confidence=1.5, explanation="test"
        )

def test_bm25_max_15():
    claims = [Claim(claim_id=f"C{i}", text="слово", document_id="D1", evidence="ev", status="direct") for i in range(30)]
    result = bm25_retrieve("слово", claims, top_k=15)
    assert len(result) <= 15

def test_invalid_evidence_id():
    # Проверка, что evidence_ids существуют в claims (логика в evaluation.py)
    pass 