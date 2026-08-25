from __future__ import annotations

import pytest

from medical_triage.priority import assign_priority


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Paciente com dor no peito e dificuldade para respirar", "high"),
        ("The patient has persistent fever and vomiting", "medium"),
        ("Routine follow-up with no acute complaint", "low"),
    ],
)
def test_assign_priority_uses_explicit_text_rules(text: str, expected: str) -> None:
    priority, reason = assign_priority(text, "cardiovascular diseases")

    assert priority == expected
    assert "Predicted condition" in reason


def test_high_priority_wins_when_multiple_terms_match() -> None:
    priority, _ = assign_priority("High fever followed by severe bleeding", "general")
    assert priority == "high"


def test_priority_rejects_blank_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        assign_priority("  ", "general")
