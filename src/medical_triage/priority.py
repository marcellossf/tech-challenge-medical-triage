"""Transparent non-clinical business rules for the API priority field."""

from __future__ import annotations

import re
import unicodedata

HIGH_PRIORITY_TERMS = (
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "severe shortness of breath",
    "unconscious",
    "severe bleeding",
    "seizure",
    "convulsion",
    "stroke symptoms",
    "suicidal",
    "dor no peito",
    "falta de ar",
    "falta de ar intensa",
    "dificuldade para respirar",
    "inconsciente",
    "sangramento intenso",
    "convulsao",
    "sinais de avc",
    "tentativa de suicidio",
)

MEDIUM_PRIORITY_TERMS = (
    "persistent fever",
    "high fever",
    "persistent pain",
    "vomiting",
    "infection",
    "worsening",
    "febre persistente",
    "febre alta",
    "dor persistente",
    "vomito",
    "infeccao",
    "piora",
)


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def _matches(text: str, terms: tuple[str, ...]) -> list[str]:
    return [term for term in terms if re.search(rf"\b{re.escape(term)}\b", text)]


def assign_priority(text: str, predicted_label: str) -> tuple[str, str]:
    """Assign ``high``, ``medium``, or ``low`` using explicit keyword rules.

    This is a demonstrative product rule, not learned urgency, medical advice, or
    a clinically validated triage protocol. ``predicted_label`` is reported in the
    explanation but deliberately does not determine priority.
    """

    normalized_text = _normalize(text)
    if not normalized_text:
        raise ValueError("Text cannot be blank")

    high_matches = _matches(normalized_text, HIGH_PRIORITY_TERMS)
    if high_matches:
        reason = (
            f"Business rule matched high-priority term(s): {', '.join(high_matches)}. "
            f"Predicted condition {predicted_label!r} did not determine urgency."
        )
        return "high", reason

    medium_matches = _matches(normalized_text, MEDIUM_PRIORITY_TERMS)
    if medium_matches:
        reason = (
            f"Business rule matched medium-priority term(s): {', '.join(medium_matches)}. "
            f"Predicted condition {predicted_label!r} did not determine urgency."
        )
        return "medium", reason

    reason = (
        "No configured red-flag or attention term was matched; defaulted to low priority. "
        f"Predicted condition {predicted_label!r} did not determine urgency."
    )
    return "low", reason
