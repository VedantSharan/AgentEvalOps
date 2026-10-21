"""
Deterministic, rule-based prompt paraphrases (Assignment 1, Part D).

No language model is used, so the augmentation is exactly reproducible and
needs no prompt logging.  A paraphrase is built from:
  1. question-form rewrites ("How many X ...?" -> "Count how many X ....",
     "What is the X ...?" -> "Determine the X ...."),
  2. a fixed lexicon of meaning-preserving substitutions, and
  3. an optional neutral framing prefix,
with the choices driven by a seeded RNG per task.

Label preservation (rule-based): ``protected_spans`` extracts everything that
determines the gold answer - numbers, quoted strings, file names, IDs,
capitalised entity names, units and logic words (strictly, NOT, exclude,
round, equal, ...).  A paraphrase is accepted only if its protected-span
multiset equals the parent's.  A seeded sample is also exported for a human
spot check of intent.

Limitation: these are surface-level paraphrases; they test robustness to
wording and instruction framing, not to deep restructuring.
"""

from __future__ import annotations

from collections import Counter
import random
import re

FRAMES = (
    "",
    "Using the tools available to you, answer the following. ",
    "Please work out the answer to this request. ",
    "Task: ",
)

# (pattern, replacement); applied left to right; word-boundary safe.
LEXICON: tuple[tuple[str, str], ...] = (
    (r"\bCalculate\b", "Compute"),
    (r"\bcalculate\b", "compute"),
    (r"\bfind the\b", "identify the"),
    (r"\baccording to\b", "as recorded in"),
    (r"\bAccording to\b", "As recorded in"),
    (r"\bbased on\b", "using"),
    (r"\bBased on\b", "Using"),
    (r"\btotal\b", "overall"),
    (r"\bQuery the\b", "Search the"),
    (r"\bExamine\b", "Inspect"),
    (r"\bAuditing\b", "When auditing"),
    (r"\bmust be transferred\b", "needs to be transferred"),
    (r"\bcurrently serves as\b", "is currently the"),
    (r"\bcurrently leads\b", "is currently leading"),
)

_LOGIC_WORDS = r"\b(?:strictly|NOT|not|exclud(?:e|es|ed|ing)|except|without|unless|includ(?:e|es|ed|ing)|equal|greater|less|exceed(?:ed|s)?|reached|round(?:ed)?|Round|up|tie|smaller|highest|lowest|maximum|minimum|missing|blank|only|each|all|per|both|either|OR|and|or|before|after|current|superseded|active|archived)\b"


def protected_spans(text: str) -> Counter:
    """Everything that determines the answer; must be identical before and after paraphrasing."""
    spans: list[str] = []
    spans += re.findall(r"'[^']+'", text)                                  # quoted names / files
    spans += re.findall(r"\b[A-Z]{1,4}[-_]?\d+[\w-]*\b", text)             # IDs: E001, PRJ-INF-932, T1
    spans += [n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]  # numbers
    spans += re.findall(r"\b(?:USD|GB|TB|MB|Mb|Mbps|ms|hours?|minutes?|days?|%|x)\b|%", text)
    spans += [w.lower() for w in re.findall(_LOGIC_WORDS, text)]
    spans += re.findall(r"(?<![.?!]\s)(?<!^)\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b", text)  # multi-word entities
    spans += re.findall(r">=|<=|>|<|=", text)
    spans += re.findall(r"\b[A-Z]{2,}\b", text)                            # acronyms: EU, SLA, HTTP, NOT
    return Counter(spans)


def _rewrite_question(text: str) -> str:
    m = re.match(r"^How many (.+)\?$", text.strip(), flags=re.S)
    if m:
        return f"Count how many {m.group(1)}."
    m = re.match(r"^What is the (.+)\?$", text.strip(), flags=re.S)
    if m:
        return f"Determine the {m.group(1)}."
    # "In 'file.csv', how many X ...?" -> "In 'file.csv', count how many X ...."
    m = re.match(r"^(.*,\s)how many (.+)\?$", text.strip(), flags=re.S)
    if m and "." not in m.group(1).rstrip(", ")[-3:]:
        return f"{m.group(1)}count how many {m.group(2)}."
    # Multi-sentence prompts: rewrite only the final question sentence.
    m = re.match(r"^(.*[.:)]\s)(how many|what is the)(.+)\?$", text.strip(), flags=re.S | re.I)
    if m:
        head, q, rest = m.groups()
        verb = "count how many" if q.lower() == "how many" else "determine the"
        return f"{head}Please {verb}{rest}."
    return text


def paraphrase(prompt: str, seed: int) -> tuple[str, dict]:
    """Return (paraphrased prompt, record of operations applied)."""
    rng = random.Random(seed)
    ops: list[str] = []
    text = prompt
    if rng.random() < 0.8:
        new = _rewrite_question(text)
        if new != text:
            ops.append("question_rewrite")
            text = new
    for pat, rep in LEXICON:
        if re.search(pat, text) and rng.random() < 0.7:
            text = re.sub(pat, rep, text)
            ops.append(f"lexicon:{rep}")
    frame = FRAMES[rng.randrange(len(FRAMES))]
    if frame:
        text = frame + text
        ops.append(f"frame:{frame.strip()}")
    if text == prompt:  # guarantee a change: force the first applicable lexicon rule
        for pat, rep in LEXICON:
            if re.search(pat, text):
                text = re.sub(pat, rep, text, count=1)
                ops.append(f"forced_lexicon:{rep}")
                break
        else:
            text = FRAMES[1] + text
            ops.append(f"forced_frame:{FRAMES[1].strip()}")
    return text, {"operations": ops, "seed": seed}


def spans_preserved(original: str, para: str) -> tuple[bool, str]:
    a, b = protected_spans(original), protected_spans(para)
    if a == b:
        return True, f"{sum(a.values())} protected spans identical"
    return False, f"lost={dict(a - b)} added={dict(b - a)}"
