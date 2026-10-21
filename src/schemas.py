from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

class Document(BaseModel):
    document_id: str
    title: str
    source_type: str
    created_at: str
    author: str
    text: str

class Claim(BaseModel):
    claim_id: str
    text: str
    document_id: str
    evidence: str
    status: Literal["direct", "reported", "inferred", "disputed"]

class Question(BaseModel):
    question_id: str
    question: str
    gold_status: Literal["entailed", "contradicted", "both", "insufficient"]
    gold_answer: str
    gold_evidence_ids: List[str]
    gold_counterevidence_ids: List[str]

class ModelOutput(BaseModel):
    question_id: str
    status: Literal["entailed", "contradicted", "both", "insufficient"]
    answer: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    counterevidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

    @model_validator(mode='after')
    def validate_constraints(self):
        if self.status == "both" and (not self.evidence_ids or not self.counterevidence_ids):
            raise ValueError("При статусе 'both' должны быть заполнены и evidence_ids, и counterevidence_ids")
        if self.status == "insufficient" and self.answer:
            if "недостаточно" not in self.answer.lower() and "null" not in str(self.answer).lower():
                pass # Допускается, если модель объясняет, чего не хватает
        return self