from __future__ import annotations

import glob
import html
import json
import os
import random
import re
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))
AMI_DIR = os.path.join(ROOT, "data", "raw", "ami", "annotations")
QMSUM_DIR = os.path.join(ROOT, "data", "raw", "qmsum", "data")

WORD_RE = re.compile(r'<(\w+)\s+nite:id="[\w.]+?words(\d+)"([^>/]*)(?:/>|>(.*?)</\1>)', re.S)
DACT_RE = re.compile(r'<dact nite:id="([^"]+)"[^>]*>(.*?)</dact>', re.S)
DECISION_RE = re.compile(r'<decision nite:id="[^"]*">(.*?)</decision>', re.S)
SPAN_RE = re.compile(r'href="[\w.]+\.([A-Z])\.words\.xml#id\([\w.]+?words(\d+)\)(?:\.\.id\([\w.]+?words(\d+)\))?"')


@dataclass
class Turn:
    meeting: str
    speaker: str
    turn_id: str
    text: str
    start: float | None
    end: float | None
    label: int = field(default=0)


def _read(path: str) -> str:
    return open(path, encoding="iso-8859-1").read()


def parse_words_xml(path: str) -> dict[int, dict]:
    out = {}
    for tag, wid, attrs, inner in WORD_RE.findall(_read(path)):
        st = re.search(r'starttime="([\d.]+)"', attrs)
        et = re.search(r'endtime="([\d.]+)"', attrs)
        out[int(wid)] = {
            "text": html.unescape(inner.strip()) if tag == "w" and inner else "",
            "start": float(st.group(1)) if st else None,
            "end": float(et.group(1)) if et else None,
        }
    return out


def parse_dialogue_acts(path: str) -> list[tuple[str, int, int]]:
    turns = []
    for dact_id, body in DACT_RE.findall(_read(path)):
        ids = [int(x) for x in re.findall(r"words(\d+)", body)]
        if ids:
            turns.append((dact_id, min(ids), max(ids)))
    return turns


def parse_decision_spans(path: str) -> dict[str, list[tuple[int, int]]]:
    spans: dict[str, list[tuple[int, int]]] = {}
    for body in DECISION_RE.findall(_read(path)):
        for speaker, start, end in SPAN_RE.findall(body):
            spans.setdefault(speaker, []).append((int(start), int(end or start)))
    return spans


def _overlaps(a0, a1, b0, b1) -> bool:
    return a0 <= b1 and b0 <= a1


def list_decision_meetings() -> list[str]:
    files = glob.glob(os.path.join(AMI_DIR, "decision", "manual", "*.decision.xml"))
    return sorted(os.path.basename(f).replace(".decision.xml", "") for f in files)


def build_ami_turns(meeting: str) -> list[Turn]:
    decision_path = os.path.join(AMI_DIR, "decision", "manual", f"{meeting}.decision.xml")
    spans = parse_decision_spans(decision_path) if os.path.exists(decision_path) else {}

    turns: list[Turn] = []
    for dact_path in glob.glob(os.path.join(AMI_DIR, "dialogueActs", f"{meeting}.*.dialog-act.xml")):
        if "adjacency-pairs" in dact_path:
            continue
        speaker = os.path.basename(dact_path).split(".")[1]
        words_path = os.path.join(AMI_DIR, "words", f"{meeting}.{speaker}.words.xml")
        if not os.path.exists(words_path):
            continue
        words = parse_words_xml(words_path)
        sp_spans = spans.get(speaker, [])

        for turn_id, w0, w1 in parse_dialogue_acts(dact_path):
            text = " ".join(words[i]["text"] for i in range(w0, w1 + 1) if i in words and words[i]["text"])
            if not text:
                continue
            start = next((words[i]["start"] for i in range(w0, w1 + 1) if i in words and words[i]["start"] is not None), None)
            end = next((words[i]["end"] for i in range(w1, w0 - 1, -1) if i in words and words[i]["end"] is not None), None)
            label = int(any(_overlaps(w0, w1, s0, s1) for s0, s1 in sp_spans))
            turns.append(Turn(meeting, speaker, turn_id, text, start, end, label))

    turns.sort(key=lambda t: (t.start is None, t.start))
    return turns


def build_decision_dataset(
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    neg_pos_ratio: float = 2.0,
    max_train_examples: int = 4000,
    seed: int = 42,
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    meetings = list_decision_meetings()
    rng.shuffle(meetings)
    n_val = max(1, round(len(meetings) * val_frac))
    n_test = max(1, round(len(meetings) * test_frac))
    val_meetings = set(meetings[:n_val])
    test_meetings = set(meetings[n_val:n_val + n_test])
    train_meetings = set(meetings[n_val + n_test:])

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for meeting in meetings:
        bucket = "val" if meeting in val_meetings else "test" if meeting in test_meetings else "train"
        for t in build_ami_turns(meeting):
            splits[bucket].append({"text": t.text, "label": t.label, "meeting": t.meeting})

    pos = [r for r in splits["train"] if r["label"] == 1]
    neg = [r for r in splits["train"] if r["label"] == 0]
    rng.shuffle(neg)
    neg = neg[: int(len(pos) * neg_pos_ratio)]
    train = pos + neg
    rng.shuffle(train)
    if len(train) > max_train_examples:
        train = train[:max_train_examples]
    splits["train"] = train
    return splits


def load_qmsum(domain: str = "ALL", split: str = "test") -> list[dict]:
    records = []
    for path in glob.glob(os.path.join(QMSUM_DIR, domain, split, "*.json")):
        with open(path, encoding="utf-8") as f:
            records.append(json.load(f))
    return records


if __name__ == "__main__":
    ds = build_decision_dataset()
    for name, rows in ds.items():
        pos = sum(r["label"] for r in rows)
        print(f"{name}: {len(rows)} turns, {pos} positive ({pos / max(len(rows), 1):.1%})")
