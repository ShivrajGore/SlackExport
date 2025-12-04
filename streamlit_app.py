from __future__ import annotations

import json
import zipfile
from datetime import date, datetime, time as dt_time, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

from src.firestore_store import FirestoreStore
from src.models import AppConfig
from src.slack_etl import run_pipeline
from src.slack_single_export import export_single_thread

st.set_page_config(
    page_title="Slack Knowledge Export",
    layout="wide",
    page_icon=":robot_face:",
)
LOCAL_TZ = datetime.now().astimezone().tzinfo

PRIMARY = "#5C6CFF"
SECONDARY = "#1C1F33"
ACCENT = "#22D3EE"
CARD_BG = "#14172B"
TEXT_LIGHT = "#F5F7FF"


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: radial-gradient(circle at top, #12172b 0%, #080912 45%, #05060f 100%);
                color: {TEXT_LIGHT};
                font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
            }}
            h1, h2, h3, h4, h5, h6, .stMetric label {{
                color: {TEXT_LIGHT};
            }}
            .custom-card {{
                border-radius: 18px;
                padding: 1.25rem 1.5rem;
                background: {CARD_BG};
                border: 1px solid rgba(255,255,255,0.05);
                box-shadow: 0 20px 45px rgba(0,0,0,0.35);
            }}
            .stButton>button {{
                background: linear-gradient(120deg, {PRIMARY}, {ACCENT});
                color: white;
                border: none;
                border-radius: 999px;
                padding: 0.45rem 1.2rem;
                font-weight: 600;
                box-shadow: 0 10px 25px rgba(92,108,255,0.3);
                width: auto;
                min-width: 200px;
                max-width: 320px;
            }}
            .stButton>button:disabled {{
                background: rgba(255,255,255,0.1);
                color: rgba(255,255,255,0.6);
            }}
            .css-1dp5vir, .css-1cypcdb, .st-emotion-cache-1r6slb0 {{
                background: transparent;
            }}
            .metric-card {{
                border-radius: 16px;
                background: rgba(255,255,255,0.03);
                padding: 1rem 1.2rem;
                border: 1px solid rgba(255,255,255,0.05);
            }}
            .metric-value {{
                font-size: 2rem;
                font-weight: 700;
                color: {ACCENT};
            }}
            .metric-label {{
                font-size: 0.9rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: rgba(255,255,255,0.7);
            }}
            .stDataFrame thead {{
                background: rgba(255,255,255,0.04);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_store() -> FirestoreStore:
    if "store" not in st.session_state:
        st.session_state.store = FirestoreStore()
    return st.session_state.store


def load_config(store: FirestoreStore) -> AppConfig:
    if "config" not in st.session_state:
        st.session_state.config = store.fetch_config()
    return st.session_state.config


def save_config(store: FirestoreStore, config: AppConfig) -> None:
    store.save_config(config)
    st.session_state.config = config


def parse_channels(raw_value: str) -> List[str]:
    normalized = raw_value.replace(",", "\n")
    return [item.strip() for item in normalized.splitlines() if item.strip()]


def combine_date(date_value: date, end_of_day: bool = False) -> datetime:
    tz = LOCAL_TZ or datetime.now().astimezone().tzinfo
    if end_of_day:
        date_value = date_value + timedelta(days=1)
    dt = datetime.combine(date_value, dt_time.min, tzinfo=tz)
    return dt


def trigger_pipeline(store: FirestoreStore, start_dt=None, end_dt=None) -> None:
    st.session_state.is_exporting = True
    progress_container = st.empty()
    progress_bar = progress_container.progress(0, text="Preparing export...")

    def update_progress(current: int, total: int) -> None:
        if total <= 0:
            progress_bar.progress(0, text="Preparing export...")
            return
        percent = min(100, max(0, int((current / total) * 100)))
        progress_bar.progress(percent, text=f"Processing chunk {current}/{total}")

    with st.spinner("Running export..."):
        try:
            result = run_pipeline(
                store,
                start_time=start_dt,
                end_time=end_dt,
                progress_callback=update_progress,
            )
            result["selected_files"] = None
            result["selected_records"] = None
            result["export_type"] = "batch"
            st.session_state.last_result = result
            st.session_state.status_message = (
                "success",
                (
                    "Export completed. "
                    f"Threads exported: {result['threads_exported']}. "
                    f"Failures: {len(result['failures'])}"
                ),
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.session_state.status_message = ("error", str(exc))
        finally:
            progress_container.empty()
            st.session_state.is_exporting = False


def render_logs(store: FirestoreStore) -> None:
    logs = store.fetch_logs(limit=50)
    if not logs:
        st.info("No logs recorded yet.")
        return
    st.dataframe(logs, use_container_width=True)


def trigger_single_thread(store: FirestoreStore, permalink: str) -> None:
    permalink = permalink.strip()
    if not permalink:
        st.warning("Enter a Slack thread permalink first.")
        return
    config: AppConfig = st.session_state.get("config") or store.fetch_config()
    st.session_state.is_exporting = True
    with st.spinner("Exporting thread..."):
        try:
            entry, record = export_single_thread(config, permalink)
            ts_value = float(entry.source_ts)
            fallback_used = entry.summary_provider.endswith("transcript")
            result = {
                "threads_exported": 1,
                "summaries_generated": 0 if fallback_used else 1,
                "summaries_fallback": 1 if fallback_used else 0,
                "embedding_records": 1,
                "channel_stats": {entry.source_channel: 1},
                "failures": [],
                "summary_issues": [],
                "summary_providers": {entry.summary_provider: 1},
                "manual_range": True,
                "start_timestamp": ts_value,
                "end_timestamp": ts_value + 0.000001,
                "output_dir": config.knowledge_base_dir,
                "selected_files": [str(entry.file_path)] if entry.file_path else None,
                "selected_records": [record],
                "export_type": "single",
            }
            st.session_state.last_result = result
            st.session_state.status_message = (
                "success",
                f"Thread exported to {entry.file_path}",
            )
        except Exception as exc:  # pylint: disable=broad-except
            st.session_state.status_message = ("error", str(exc))
        finally:
            st.session_state.is_exporting = False


def _normalize_ts(value) -> Optional[float]:
    try:
        if value is None:
            return None
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num <= 0:
        return None
    return num


def format_range_label(
    start_ts: Optional[float],
    end_ts: Optional[float],
    manual_range: bool,
) -> str:
    tz = LOCAL_TZ or datetime.now().astimezone().tzinfo
    start = _normalize_ts(start_ts)
    end = _normalize_ts(end_ts)
    if start and end:
        start_label = datetime.fromtimestamp(start, tz).strftime("%Y%m%d")
        end_adjusted = end - 1 if end - 1 > 0 else end
        end_label = datetime.fromtimestamp(end_adjusted, tz).strftime("%Y%m%d")
        label = start_label if start_label == end_label else f"{start_label}-{end_label}"
    elif start:
        label = datetime.fromtimestamp(start, tz).strftime("%Y%m%d")
    else:
        label = datetime.now(tz).strftime("%Y%m%d")
    return label


def build_download_bundle(
    output_dir: str,
    range_label: str,
    selected_files: Optional[List[str]] = None,
) -> Optional[Tuple[bytes, str]]:
    base_path = Path(output_dir)
    if not base_path.exists():
        return None
    if selected_files:
        files = [Path(item) for item in selected_files if Path(item).exists()]
        base_for_rel = base_path
    else:
        files = [path for path in base_path.rglob("*") if path.is_file()]
        base_for_rel = base_path
    if not files:
        return None
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            try:
                arcname = file_path.relative_to(base_for_rel)
            except ValueError:
                arcname = file_path.name
            archive.write(file_path, arcname=str(arcname))
    buffer.seek(0)
    filename = f"threads_{range_label}.zip"
    return buffer.getvalue(), filename


def build_embeddings_payload(
    output_dir: str,
    start_ts: Optional[float],
    end_ts: Optional[float],
    range_label: str,
    selected_records: Optional[List[dict]] = None,
) -> Optional[Tuple[bytes, str]]:
    start = _normalize_ts(start_ts)
    end = _normalize_ts(end_ts)
    lines: List[str] = []
    if selected_records:
        lines = [json.dumps(record, ensure_ascii=False) for record in selected_records]
    else:
        embeddings_path = Path(output_dir) / "embeddings" / "threads.jsonl"
        if not embeddings_path.exists():
            return None
        with embeddings_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_value = _normalize_ts(record.get("root_ts"))
                if start is not None and (ts_value is None or ts_value < start):
                    continue
                if end is not None and (ts_value is None or ts_value >= end):
                    continue
                lines.append(json.dumps(record, ensure_ascii=False))
    if not lines:
        return None
    payload = "\n".join(lines).encode("utf-8")
    filename = f"threads_{range_label}.jsonl"
    return payload, filename


def render_metrics(last_result: dict) -> None:
    metrics = [
        ("Threads Exported", last_result.get("threads_exported", 0)),
        ("LLM Summaries", last_result.get("summaries_generated", 0)),
        ("Transcript Fallbacks", last_result.get("summaries_fallback", 0)),
        ("Embedding Chunks", last_result.get("embedding_records", 0)),
    ]
    cols = st.columns(len(metrics))
    for (label, value), col in zip(metrics, cols):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_downloads(last_result: dict) -> None:
    output_dir = last_result.get("output_dir")
    if not output_dir:
        st.caption("No export directory available for download.")
        return
    range_label = format_range_label(
        last_result.get("start_timestamp"),
        last_result.get("end_timestamp"),
        last_result.get("manual_range", False),
    )
    selected_files = last_result.get("selected_files")
    selected_records = last_result.get("selected_records")
    bundle = build_download_bundle(output_dir, range_label, selected_files=selected_files)
    embeddings_payload = build_embeddings_payload(
        output_dir,
        last_result.get("start_timestamp"),
        last_result.get("end_timestamp"),
        range_label,
        selected_records=selected_records,
    )
    col_zip, col_json = st.columns(2)
    with col_zip:
        if bundle:
            data, filename = bundle
            st.download_button(
                "⬇️ Download Threads (.zip)",
                data=data,
                file_name=filename,
                mime="application/zip",
            )
        else:
            st.caption("No thread files available for download.")
    with col_json:
        if embeddings_payload:
            data, filename = embeddings_payload
            st.download_button(
                "⬇️ Download Embeddings (.jsonl)",
                data=data,
                file_name=filename,
                mime="application/json",
            )
        else:
            st.caption("No embedding records found for this export.")


def main() -> None:
    inject_theme()
    store = get_store()
    config = load_config(store)

    st.markdown(
        """
        <div class="custom-card">
            <h1 style="margin-bottom:0.3rem;">Slack Knowledge Pilot</h1>
            <p style="margin-top:0; color: rgba(245,247,255,0.75);">
                Seamlessly distill Slack incidents into CustomGPT-ready knowledge.
                Tag threads, export transcripts, and keep your AI agents current.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.subheader("Control Center", divider="rainbow")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button(
                "⚡ Run Incremental Export",
                disabled=st.session_state.get("is_exporting", False),
            ):
                trigger_pipeline(store)
        with col2:
            today = date.today()
            default_start = today
            start_date = st.date_input("Start Date", value=default_start)
            end_date = st.date_input("End Date", value=today)
            if st.button(
                "📅 Run Manual Export",
                disabled=st.session_state.get("is_exporting", False),
            ):
                start_dt = combine_date(start_date, end_of_day=False)
                end_dt = combine_date(end_date, end_of_day=True)
                trigger_pipeline(store, start_dt=start_dt, end_dt=end_dt)
        st.divider()
        col_link, col_link_btn = st.columns([3, 1])
        with col_link:
            permalink = st.text_input(
                "Paste a Slack thread permalink",
                placeholder="https://yourworkspace.slack.com/archives/…",
            )
        with col_link_btn:
            if st.button(
                "🎯 Export Thread",
                disabled=st.session_state.get("is_exporting", False),
            ):
                trigger_single_thread(store, permalink)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Configuration", expanded=False):
        st.markdown(
            "Use your OpenAI enterprise key as the primary summarizer. Gemini remains a secondary fallback."
        )
        slack_token = st.text_input(
            "Slack Bot Token",
            value=config.slack_token,
            type="password",
        )
        channels_raw = st.text_area(
            "Channel IDs (one per line)",
            value="\n".join(config.channel_ids),
        )
        knowledge_dir = st.text_input(
            "Knowledge Base Directory",
            value=config.knowledge_base_dir,
        )
        openai_key = st.text_input(
            "OpenAI API Key",
            value=config.openai_api_key,
            type="password",
        )
        openai_model = st.text_input(
            "OpenAI Model",
            value=config.openai_model,
        )
        gemini_key = st.text_input(
            "Gemini API Key",
            value=config.gemini_key,
            type="password",
        )
        gemini_model = st.text_input(
            "Gemini Model",
            value=config.gemini_model,
        )
        if st.button("💾 Save Configuration", use_container_width=True):
            new_config = AppConfig(
                slack_token=slack_token.strip(),
                gemini_key=gemini_key.strip(),
                openai_api_key=openai_key.strip(),
                channel_ids=parse_channels(channels_raw),
                knowledge_base_dir=knowledge_dir.strip() or "knowledge_base",
                gemini_model=gemini_model.strip() or config.gemini_model,
                openai_model=openai_model.strip() or config.openai_model,
            )
            save_config(store, new_config)
            st.success("Configuration saved.")

    status = st.session_state.get("status_message")
    if status:
        level, message = status
        getattr(st, level)(message)
    if st.session_state.get("last_result"):
        last_result = st.session_state["last_result"]
        st.json(last_result)
        render_metrics(last_result)
        render_downloads(last_result)
        provider_stats = last_result.get("summary_providers") or {}
        if provider_stats:
            with st.expander("Summary providers breakdown"):
                st.write(provider_stats)
        summary_issues = last_result.get("summary_issues") or []
        if summary_issues:
            with st.expander("Summary issues (fallbacks and provider errors)"):
                for issue in summary_issues:
                    st.write(f"- {issue}")

    st.subheader("Execution Logs")
    render_logs(store)


if __name__ == "__main__":
    main()
