from __future__ import annotations
from typing import List, Tuple

import scispacy            
import scispacy.linking   

import spacy
import en_core_sci_md
from scispacy.abbreviation import AbbreviationDetector

_NLP = None


def _get_nlp(score_threshold: float = 0.85) -> spacy.language.Language:
    """
    Build (once) and return a SciSpaCy pipeline with abbreviation expansion + UMLS linker.
    """
    global _NLP
    if _NLP is not None:
        return _NLP

    nlp = en_core_sci_md.load()

    if "abbreviation_detector" not in nlp.pipe_names:
        try:
            nlp.add_pipe("abbreviation_detector")
        except Exception:
            nlp.add_pipe(AbbreviationDetector(nlp))

    # UMLS linker
    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe(
            "scispacy_linker",
            config={
                "linker_name": "umls",
                "resolve_abbreviations": True,
                "threshold": score_threshold,
            },
        )

    _NLP = nlp
    return _NLP


def _build_tagged_text(text: str, spans: List[Tuple[int, int, str]]) -> str:
    """
    Replace character spans in `text` with provided replacements. Spans must be non-overlapping.
    """
    if not spans:
        return text
    spans = sorted(spans, key=lambda x: x[0])

    out = []
    cursor = 0
    for start, end, rep in spans:
        if start < cursor:
            continue
        out.append(text[cursor:start])
        out.append(rep)
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def umls_expand(
    text: str,
    top_k: int = 1,
    score_threshold: float = 0.85,
    tag_template: str = "{text}{{{cui}|{name}}}",
) -> str:
    """
    Convert input text into a UMLS-tagged string by wrapping linked entities with tags.

    Parameters
    ----------
    text : str
        Raw input text.
    top_k : int
        How many candidate CUIs to include per entity (1 = best only).
    score_threshold : float
        Minimum linker confidence to keep a candidate.
    tag_template : str
        Format for replacement. You can change it; available fields: {text}, {cui}, {name}, {score}.

    Returns
    -------
    str
        Text with entities replaced by a tag like:
        original{CUI|Preferred Name}
    """
    nlp = _get_nlp(score_threshold)
    doc = nlp(text)

    linker = nlp.get_pipe("scispacy_linker")

    replacements: List[Tuple[int, int, str]] = []

    for ent in doc.ents:
        cands = [(cui, score) for cui, score in (ent._.kb_ents or []) if score >= score_threshold]
        if not cands:
            continue

        for i, (cui, score) in enumerate(cands[:max(1, top_k)]):
            kb_ent = linker.kb.cui_to_entity.get(cui)
            if not kb_ent:
                continue
            name = kb_ent.canonical_name
            rep = tag_template.format(text=ent.text, cui=cui, name=name, score=score)
            if i == 0:
                primary_rep = rep
            else:
                primary_rep += f";{cui}|{name}"

        replacements.append((ent.start_char, ent.end_char, primary_rep))

    return _build_tagged_text(text, replacements)


if __name__ == "__main__":
    demo = "The patient has atrial fibrillation (AF) and was given ASA."
    print(umls_expand(demo))