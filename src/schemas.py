"""Validated data contracts."""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

Status = Literal['entailed', 'contradicted', 'both', 'insufficient']

class Record(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Document(Record):
    document_id: str
    title: str
    source_type: str
    created_at: datetime
    author: str
    text: str

class Claim(Record):
    claim_id: str
    text: str
    document_id: str
    evidence: str
    status: Literal['direct', 'reported', 'inferred', 'disputed']
    event_at: datetime | None = None

class Question(Record):
    question_id: str
    question: str
    question_type: Literal['direct', 'multi_hop', 'temporal', 'rule_exception', 'contradiction', 'insufficient']
    gold_status: Status
    gold_answer: str
    gold_evidence_ids: list[str]
    gold_counterevidence_ids: list[str] = Field(default_factory=list)

class ModelOutput(Record):
    question_id: str
    status: Status
    answer: str | None
    evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1)
    missing_information: str | None = None

    @model_validator(mode='after')
    def validate_constraints(self, info: ValidationInfo):
        if self.status == 'both' and (not self.evidence_ids or not self.counterevidence_ids):
            raise ValueError('both requires evidence and counterevidence')
        if set(self.evidence_ids) & set(self.counterevidence_ids):
            raise ValueError('One claim cannot support both sides')
        if self.status == 'insufficient' and (self.answer is not None or not self.missing_information):
            raise ValueError('insufficient requires answer=null and missing_information')
        if self.status != 'insufficient' and not self.answer:
            raise ValueError('A supported status requires an answer')
        context = info.context or {}
        allowed = context.get('allowed_ids')
        if allowed is not None and not set(self.evidence_ids + self.counterevidence_ids) <= set(allowed):
            raise ValueError('Evidence ID is not available in this context')
        if context.get('question_id', self.question_id) != self.question_id:
            raise ValueError('Wrong question_id')
        return self

class StressCase(Record):
    case_id: str
    question_id: str
    kind: Literal['permutation', 'remove_premise', 'add_contradiction']
    remove_claim_id: str | None = None
    added_document: Document | None = None
    added_claim: Claim | None = None
    expected_status: Status
    expected_evidence_ids: list[str]
    expected_counterevidence_ids: list[str] = Field(default_factory=list)
    rationale: str
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_digest: str | None = None

    @model_validator(mode='after')
    def one_factor(self):
        if self.kind == 'remove_premise':
            if not self.remove_claim_id or self.added_document or self.added_claim:
                raise ValueError('Removal must change only one premise')
        elif self.kind == 'add_contradiction':
            if self.remove_claim_id or not self.added_document or not self.added_claim:
                raise ValueError('Addition requires one source and one claim')
        elif self.remove_claim_id or self.added_document or self.added_claim:
            raise ValueError('Permutation must not change content')
        return self
