from __future__ import annotations

import os
import re
from functools import lru_cache

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "decision_classifier")

COMMITMENT_RE = re.compile(
    r"\b(I('| a)?ll|I will|I('m| am) going to|I need to|I('ve| have) to|let me)\b", re.I
)
DEADLINE_RE = re.compile(
    r"\b(today|tomorrow|tonight|this week|next week|by \w+day|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|end of (the )?(week|month|day))\b",
    re.I,
)


def score_turns(turns: list[dict], n: int = 5, min_content_words: int = 5) -> list[dict]:
    freq: dict[str, int] = {}
    for t in turns:
        for w in re.findall(r"[a-zA-Z']+", t["text"].lower()):
            if w not in ENGLISH_STOP_WORDS:
                freq[w] = freq.get(w, 0) + 1

    def content_words(t):
        return [w for w in re.findall(r"[a-zA-Z']+", t["text"].lower()) if w not in ENGLISH_STOP_WORDS]

    candidates = [i for i, t in enumerate(turns) if len(content_words(t)) >= min_content_words]
    if not candidates:
        candidates = list(range(len(turns)))

    raw_scores = [sum(freq.get(w, 0) for w in content_words(t)) for t in turns]
    top_n = set(sorted(candidates, key=lambda i: raw_scores[i], reverse=True)[:n])
    max_score = max(raw_scores) or 1

    return [
        {
            "text": t["text"],
            "speaker": t.get("speaker"),
            "start": t.get("start"),
            "turn_id": t.get("turn_id"),
            "score": round(raw_scores[i] / max_score, 3),
            "selected": i in top_n,
        }
        for i, t in enumerate(turns)
    ]


def summarize_extractive(turns: list[dict], n: int = 5, min_content_words: int = 5) -> str:
    scored = score_turns(turns, n=n, min_content_words=min_content_words)
    return " ".join(s["text"] for s in scored if s["selected"])


ABSTRACTIVE_MODEL = "philschmid/bart-large-cnn-samsum"


@lru_cache(maxsize=1)
def _abstractive_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ABSTRACTIVE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(ABSTRACTIVE_MODEL)
    return tokenizer, model


def chunk_turns(turns: list[dict], max_chars: int = 3000) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_len = 0
    for t in turns:
        t_len = len(t["text"]) + 1
        if current and current_len + t_len > max_chars:
            chunks.append(current)
            current, current_len = [], 0
        current.append(t)
        current_len += t_len
    if current:
        chunks.append(current)
    return chunks


def _generate(text: str, max_length: int = 120, min_length: int = 30) -> str:
    tokenizer, model = _abstractive_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_len = inputs["input_ids"].shape[1]
    min_length = min(min_length, max(5, input_len // 3))
    max_length = max(max_length, min_length + 10)
    output_ids = model.generate(**inputs, max_length=max_length, min_length=min_length, num_beams=4)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def summarize_chunk(chunk: list[dict]) -> str:
    text = " ".join(t["text"] for t in chunk)
    return _generate(text) if text.strip() else ""


def summarize_abstractive(turns: list[dict], max_chars: int = 3000) -> list[str]:
    return [s for c in chunk_turns(turns, max_chars=max_chars) if (s := summarize_chunk(c))]


def summarize_overall(section_summaries: list[str]) -> str:
    if not section_summaries:
        return ""
    if len(section_summaries) == 1:
        return section_summaries[0]
    return _generate(" ".join(section_summaries), max_length=150, min_length=40)


LABEL_MODEL = "google/flan-t5-base"
LABEL_FEWSHOT = """Rewrite each item as a short 2-4 word title. No punctuation.

Item: I will send the finance report to the team by Friday.
Title: Send finance report

Item: I need to email the vendor about API access.
Title: Email vendor

Item: We decided to postpone the launch until next week.
Title: Postpone launch

Item: {text}
Title:"""

_FILLER_LEAD_RE = re.compile(
    r"^(yes|yeah|sure|okay|ok|well|so|right|great|good|alright|um+|uh+|and|but|also|now|then)[,.]?\s+", re.I
)
_COMMITMENT_LEAD_RE = re.compile(
    r"^(i('| a)?ll|i will|i('m| am) going to|i need to|i('ve| have) to|let me|"
    r"we('ve| have) decided to|we decided to|we('re| are) going to|we agreed to|"
    r"we('ll| will)|it('s| is) decided that)\s+",
    re.I,
)
_CLAUSE_SPLIT_RE = re.compile(
    r",\s+it('s| is| was)\b|,\s+i('ll| will| need)\b|"
    r"(?<!\btry)(?<!\bcome)(?<!\bgo)(?<!\btries)(?<!\bcomes)(?<!\bgoes)\s+\band\b\s+",
    re.I,
)


def _core_clause(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = _FILLER_LEAD_RE.sub("", text).strip()
        text = _COMMITMENT_LEAD_RE.sub("", text).strip()
    return _CLAUSE_SPLIT_RE.split(text)[0]


_OBJECT_DEPS = ("dobj", "obj", "attr", "oprd", "acomp", "dative", "nsubjpass")


def _span_text(doc, token):
    idxs = [t.i for t in token.subtree]
    return doc[min(idxs): max(idxs) + 1].text


def _core_phrase(text: str) -> str:
    doc = _ner_model()(text)
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None or root.pos_ not in ("VERB", "AUX"):
        content = [t.text for t in doc if not t.is_stop and t.is_alpha][:4]
        return " ".join(content) if content else text[:30]

    obj = next((c for c in root.children if c.dep_ in _OBJECT_DEPS), None)
    if obj is not None:
        return f"{root.lemma_} {_span_text(doc, obj)}"

    prep = next((c for c in root.children if c.dep_ == "prep"), None)
    if prep is not None:
        pobj = next((c for c in prep.children if c.dep_ == "pobj"), None)
        if pobj is not None:
            return f"{root.lemma_} {prep.text} {_span_text(doc, pobj)}"

    conj = next((c for c in root.children if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX")), None)
    if conj is not None:
        conj_obj = next((c for c in conj.children if c.dep_ in _OBJECT_DEPS), None)
        if conj_obj is not None:
            return f"{conj.lemma_} {_span_text(doc, conj_obj)}"
        return conj.lemma_

    return root.lemma_


@lru_cache(maxsize=1)
def _label_model():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(LABEL_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LABEL_MODEL)
    return tokenizer, model


def short_label(text: str, max_words: int = 4) -> str:
    phrase = _core_phrase(_core_clause(text)) or text
    tokenizer, model = _label_model()
    inputs = tokenizer(LABEL_FEWSHOT.format(text=phrase), return_tensors="pt")
    output_ids = model.generate(**inputs, max_new_tokens=8, num_beams=4)
    label = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip(" \"'.")
    if not label or re.search(r"\brewrite\b|\btitle\b|\bitem\b", label, re.I):
        label = phrase
    words = label.split()[:max_words]
    return " ".join(words).capitalize() if words else text[:40]


NOT_FOUND = "Could not find in the transcript provided."

QA_PROMPT = """Answer the question using ONLY the meeting transcript below. Be concise. \
If the answer is not in the transcript, respond with exactly: {not_found}

Transcript:
{context}

Question: {question}
Answer:"""


def _relevant_turns(turns: list[dict], question: str, top_k: int = 8) -> list[dict]:
    q_words = {w for w in re.findall(r"[a-zA-Z']+", question.lower()) if w not in ENGLISH_STOP_WORDS}
    if not q_words:
        return []
    scored = []
    for t in turns:
        t_words = set(re.findall(r"[a-zA-Z']+", t["text"].lower()))
        if t.get("speaker"):
            t_words |= set(re.findall(r"[a-zA-Z']+", t["speaker"].lower()))
        overlap = len(q_words & t_words)
        if overlap:
            scored.append((overlap, t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:top_k]]


def answer_question(turns: list[dict], question: str, top_k: int = 8) -> str:
    relevant = _relevant_turns(turns, question, top_k=top_k)
    if not relevant:
        return NOT_FOUND

    context = "\n".join(f"{t.get('speaker') or 'Speaker'}: {t['text']}" for t in relevant)
    tokenizer, model = _label_model()
    prompt = QA_PROMPT.format(not_found=NOT_FOUND, context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = model.generate(**inputs, max_new_tokens=60, num_beams=4)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    if not answer or re.search(r"not found|not in the transcript|cannot find|no answer|don't know", answer, re.I):
        return NOT_FOUND
    return answer


WHISPER_MODEL = "base"


@lru_cache(maxsize=1)
def _whisper_model():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")


def transcribe_audio(audio_path: str) -> list[dict]:
    model = _whisper_model()
    segments, _info = model.transcribe(audio_path, beam_size=1)
    turns = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if text:
            turns.append({"text": text, "turn_id": f"seg{i}", "start": seg.start, "end": seg.end, "speaker": None})
    return turns


@lru_cache(maxsize=1)
def _decision_pipeline():
    if not os.path.isdir(MODEL_DIR):
        return None
    from transformers import pipeline
    return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)


def classify_decisions(turns: list[dict]) -> list[dict]:
    clf = _decision_pipeline()
    if clf is None:
        raise RuntimeError("No fine-tuned classifier found. Run train_classifier.py first.")

    results = []
    texts = [t["text"] for t in turns]
    for turn, pred in zip(turns, clf(texts, truncation=True, max_length=32)):
        label = int(pred["label"].endswith("1"))
        if label:
            results.append({
                "decision": short_label(turn["text"]),
                "reference": turn["text"],
                "confidence": round(pred["score"], 3),
                "evidence_turn_id": turn.get("turn_id"),
                "start": turn.get("start"),
                "speaker": turn.get("speaker"),
            })
    return results


@lru_cache(maxsize=1)
def _ner_model():
    import spacy
    return spacy.load("en_core_web_sm")


def extract_action_items(turns: list[dict]) -> list[dict]:
    nlp = _ner_model()
    items = []
    for t in turns:
        text = t["text"]
        commit = COMMITMENT_RE.search(text)
        if not commit:
            continue
        doc = nlp(text)
        deadline_match = DEADLINE_RE.search(text)
        people = [e.text for e in doc.ents if e.label_ == "PERSON"]
        dates = [e.text for e in doc.ents if e.label_ == "DATE"]

        owner = t.get("speaker") if re.match(r"\bI\b|I'", commit.group(0), re.I) else (people[0] if people else t.get("speaker"))
        deadline = deadline_match.group(0) if deadline_match else (dates[0] if dates else None)
        confidence = 0.8 if deadline else 0.5

        items.append({
            "task": short_label(text[commit.start():]),
            "reference": text,
            "owner": owner,
            "deadline": deadline,
            "confidence": confidence,
            "evidence_turn_id": t.get("turn_id"),
            "start": t.get("start"),
        })
    return items


def run_pipeline(turns: list[dict], decisions_available: bool = True) -> dict:
    section_summaries = summarize_abstractive(turns)
    result = {
        "summary_extractive": summarize_extractive(turns),
        "summary_sections": section_summaries,
        "summary_overall": summarize_overall(section_summaries),
        "action_items": extract_action_items(turns),
        "decisions": [],
    }
    if decisions_available:
        try:
            result["decisions"] = classify_decisions(turns)
        except RuntimeError as e:
            result["decisions_error"] = str(e)
    else:
        result["decisions_error"] = "Decision detection is only trained/evaluated on AMI meetings (no gold labels for this source)."
    return result
