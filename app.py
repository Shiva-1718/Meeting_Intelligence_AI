import base64
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from pipeline import answer_question, chunk_turns, classify_decisions, extract_action_items, score_turns, summarize_chunk, summarize_overall, transcribe_audio

ASSETS = Path(__file__).parent / "assets"
st.set_page_config(page_title="Meeting Intelligence", page_icon="🎙️", layout="wide")


def _svg_b64(name: str) -> str:
    return base64.b64encode((ASSETS / name).read_bytes()).decode()


st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(160deg, #EAF3F5 0%, #F7FAFB 40%, #FDF9F2 100%);
}}
.block-container {{ padding-top: 2rem; max-width: 1050px; }}

.hero {{
    background: linear-gradient(135deg, #2D7080 0%, #1E4E5A 100%);
    color: white; padding: 2.2rem 2.5rem; border-radius: 20px; margin-bottom: 1.75rem;
    box-shadow: 0 12px 30px rgba(45,112,128,0.25);
    display: flex; align-items: center; justify-content: space-between; gap: 1.5rem;
}}
.hero-text h1 {{ margin: 0; font-size: 2rem; }}
.hero-text p {{ margin: 0.5rem 0 0; opacity: 0.92; font-size: 1.02rem; max-width: 34rem; }}
.hero img {{ height: 140px; }}

.path-card {{
    background: white; border-radius: 16px; padding: 1.4rem 1.5rem;
    box-shadow: 0 4px 16px rgba(28,43,48,0.08); border: 1px solid rgba(45,112,128,0.10);
    text-align: center; height: 100%;
}}
.path-card img {{ height: 90px; margin-bottom: 0.5rem; }}
.path-card h3 {{ color: #2D7080; margin: 0.3rem 0; }}
.path-card p {{ color: #5b6b6f; font-size: 0.88rem; margin: 0; }}

.card {{
    background: white; border-radius: 16px; padding: 1.3rem 1.5rem;
    box-shadow: 0 4px 16px rgba(28,43,48,0.08); border: 1px solid rgba(45,112,128,0.10); height: 100%;
}}
.card h4 {{ margin-top: 0; color: #2D7080; }}

[data-testid="stMetricValue"] {{ color: #2D7080; }}
div[data-testid="stStatusWidget"] {{ border-radius: 12px; }}

.stButton > button {{
    border-radius: 10px; font-weight: 600; border: 1px solid rgba(45,112,128,0.25);
}}
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, #2D7080 0%, #245a66 100%); border: none;
}}
</style>
""", unsafe_allow_html=True)


def hero(title: str, subtitle: str, image: str | None = None):
    img_html = f'<img src="data:image/svg+xml;base64,{_svg_b64(image)}" />' if image else ""
    st.markdown(f"""
    <div class="hero">
      <div class="hero-text"><h1>{title}</h1><p>{subtitle}</p></div>
      {img_html}
    </div>
    """, unsafe_allow_html=True)


def fmt_time(seconds):
    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def parse_transcript_text(text: str) -> list[dict]:
    turns = []
    for i, line in enumerate(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        speaker, sep, rest = line.partition(":")
        if sep and 0 < len(speaker.split()) <= 3 and rest.strip():
            turns.append({"text": rest.strip(), "turn_id": f"line{i}", "start": None, "speaker": speaker.strip()})
        else:
            turns.append({"text": line, "turn_id": f"line{i}", "start": None, "speaker": None})
    return turns


def transcribe_uploaded(uploaded_file) -> list[dict]:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    with st.spinner("Transcribing audio with Whisper..."):
        return transcribe_audio(tmp_path)


if "path" not in st.session_state:
    st.session_state.path = None
if "view" not in st.session_state:
    st.session_state.view = "summary"


def go(path):
    st.session_state.path = path
    st.session_state.view = "summary"
    st.session_state.pop("results", None)
    st.session_state.pop("chat_history", None)


if st.session_state.path is None:
    hero(
        "Welcome to Meeting Intelligence 👋",
        "Bring a meeting in — however you have it — and get a clear summary, the "
        "decisions that were made, and action items with owners and deadlines, "
        "every one linked back to where it was said.",
        image="hero-video-call.svg",
    )

    st.markdown("#### How would you like to bring in your meeting?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="path-card">
          <img src="data:image/svg+xml;base64,{_svg_b64('hero-work-from-home.svg')}" />
          <h3>📤 Upload</h3>
          <p>Already have a transcript file or an audio recording? Upload it and we'll take it from there.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Choose Upload", key="btn_upload", type="primary", width="stretch", on_click=go, args=("upload",))
    with col2:
        st.markdown(f"""
        <div class="path-card">
          <img src="data:image/svg+xml;base64,{_svg_b64('hero-video-call.svg')}" />
          <h3>🎥 Live Interaction</h3>
          <p>Paste in notes as you type, or record the meeting live through your microphone.</p>
        </div>
        """, unsafe_allow_html=True)
        st.button("Choose Live Interaction", key="btn_live", type="primary", width="stretch", on_click=go, args=("live",))

    st.stop()


def go_home():
    st.session_state.path = None
    st.session_state.view = "summary"
    st.session_state.pop("results", None)
    st.session_state.pop("chat_history", None)


st.button("← Back to home", on_click=go_home)

turns: list[dict] | None = None

if st.session_state.path == "upload":
    hero("📤 Upload a meeting", "Bring in a transcript file or an audio recording.")
    sub = st.radio("Upload type", ["📄 Transcript file", "🎧 Audio recording"], horizontal=True, label_visibility="collapsed")

    if sub == "📄 Transcript file":
        text_file = st.file_uploader("Transcript file (.txt)", type=["txt"])
        if text_file:
            turns = parse_transcript_text(text_file.getvalue().decode("utf-8", errors="ignore"))
    else:
        audio_file = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
        if audio_file:
            turns = transcribe_uploaded(audio_file)

else:
    hero("🎥 Live Interaction", "Paste in text as you type, or record the meeting live.")
    sub = st.radio("Live type", ["✍️ Paste text", "🔴 Record meeting"], horizontal=True, label_visibility="collapsed")

    if sub == "✍️ Paste text":
        text = st.text_area("Paste transcript (one turn per line, optionally 'Speaker: text')", height=220,
                              placeholder="Sarah: Let's get started...\nTom: I'll send the report by Friday.")
        if text.strip():
            turns = parse_transcript_text(text)
    else:
        recording = st.audio_input("Record your meeting")
        if recording:
            turns = transcribe_uploaded(recording)

if turns:
    with st.expander(f"Transcript preview — {len(turns)} turns", expanded=False):
        preview = pd.DataFrame(turns)[["speaker", "text"]]
        st.dataframe(preview, width="stretch", height=220)

    if st.button("✨ Analyze meeting", type="primary", width="stretch"):
        with st.status("Analyzing meeting...", expanded=True) as status:
            st.write(f"Scoring {len(turns)} turns for the extractive baseline...")
            extractive_scores = score_turns(turns)

            sections = chunk_turns(turns)
            n = len(sections)
            if n == 1:
                st.write("Transcript fits in one pass — generating meeting summary (transformer)...")
            else:
                st.write(f"Transcript is long — split into {n} sections to avoid truncation...")
            section_summaries = []
            for i, section in enumerate(sections, 1):
                if n > 1:
                    st.write(f"Summarizing section {i} of {n} ({len(section)} turns, transformer)...")
                section_summaries.append(summarize_chunk(section))

            if n > 1:
                st.write(f"Combining {n} sections into one meeting summary...")
            summary_overall = summarize_overall(section_summaries)

            st.write("Detecting decisions (fine-tuned classifier)...")
            try:
                decisions = classify_decisions(turns)
                decisions_error = None
            except RuntimeError as e:
                decisions, decisions_error = [], str(e)

            st.write(f"Scanning {len(turns)} turns for action-item commitment phrases + NER...")
            action_items = extract_action_items(turns)

            status.update(label=f"Done — {len(decisions)} decisions, {len(action_items)} action items found", state="complete")

        st.session_state.results = {
            "extractive_scores": extractive_scores,
            "summary_sections": section_summaries,
            "summary_overall": summary_overall,
            "decisions": decisions,
            "decisions_error": decisions_error,
            "action_items": action_items,
            "turns": turns,
        }
        st.session_state.chat_history = []

    results = st.session_state.get("results")
    if results:
        r_turns = results["turns"]
        durations = [t["start"] for t in r_turns if t.get("start") is not None]
        duration = fmt_time(max(durations) - min(durations)) if durations else "—"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Turns", len(r_turns))
        m2.metric("Duration", duration)
        m3.metric("Decisions", len(results["decisions"]))
        m4.metric("Action items", len(results["action_items"]))

        st.write("")
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("📝 Summary", width="stretch", type="primary" if st.session_state.view == "summary" else "secondary"):
            st.session_state.view = "summary"
        if b2.button("✅ Decisions", width="stretch", type="primary" if st.session_state.view == "decisions" else "secondary"):
            st.session_state.view = "decisions"
        if b3.button("📌 Action Items", width="stretch", type="primary" if st.session_state.view == "actions" else "secondary"):
            st.session_state.view = "actions"
        if b4.button("💬 Ask", width="stretch", type="primary" if st.session_state.view == "ask" else "secondary"):
            st.session_state.view = "ask"

        st.write("")

        if st.session_state.view == "summary":
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="card"><h4>🧩 Extractive baseline</h4>', unsafe_allow_html=True)
                st.caption("Every line, scored by importance — no rewriting. Checked rows were picked as the highlights.")
                edf = pd.DataFrame(results["extractive_scores"])
                st.dataframe(
                    edf[["text", "speaker", "score", "selected"]],
                    width="stretch", height=320,
                    column_config={
                        "text": "Line",
                        "speaker": "Speaker",
                        "score": st.column_config.ProgressColumn("Importance", min_value=0, max_value=1),
                        "selected": st.column_config.CheckboxColumn("In highlights"),
                    },
                )
                st.markdown("</div>", unsafe_allow_html=True)
            with col2:
                sections = results["summary_sections"]
                st.markdown('<div class="card"><h4>📝 Meeting Summary</h4>', unsafe_allow_html=True)
                if len(sections) <= 1:
                    st.caption("Generated by bart-large-cnn-samsum from the full transcript.")
                else:
                    st.caption(f"Meeting was long, so it was split into {len(sections)} sections, "
                               f"summarised separately, then combined into one summary below.")
                st.write(results["summary_overall"] or "_No output._")
                if len(sections) > 1:
                    with st.expander("See section-by-section breakdown"):
                        for i, s in enumerate(sections, 1):
                            st.markdown(f"**Section {i}**")
                            st.write(s or "_No output._")
                st.markdown("</div>", unsafe_allow_html=True)

        elif st.session_state.view == "decisions":
            if results["decisions_error"]:
                st.info(results["decisions_error"])
            elif results["decisions"]:
                df = pd.DataFrame(results["decisions"])
                df["time"] = df["start"].apply(fmt_time)
                st.dataframe(
                    df[["decision", "reference", "speaker", "confidence", "time"]],
                    width="stretch",
                    column_config={
                        "decision": "Decision",
                        "reference": "Reference (evidence)",
                        "speaker": "Speaker",
                        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                        "time": "Timestamp",
                    },
                )
            else:
                st.write("_No decisions detected._")

        elif st.session_state.view == "actions":
            if results["action_items"]:
                df = pd.DataFrame(results["action_items"])
                df["time"] = df["start"].apply(fmt_time)
                st.data_editor(
                    df[["task", "reference", "owner", "deadline", "confidence", "time"]],
                    width="stretch",
                    num_rows="dynamic",
                    column_config={
                        "task": "Task",
                        "reference": "Reference (evidence)",
                        "owner": "Owner",
                        "deadline": "Deadline",
                        "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1),
                        "time": "Timestamp",
                    },
                )
            else:
                st.write("_No action items detected._")

        else:
            st.caption("Ask a question about this meeting. Answers are grounded in the transcript — "
                       "if it wasn't said, you'll get 'Could not find in the transcript provided.'")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            question = st.chat_input("Ask about this meeting...")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.spinner("Checking the transcript..."):
                    answer = answer_question(results["turns"], question)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
