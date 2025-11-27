from __future__ import annotations

from datetime import date, datetime, time as dt_time, timedelta
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile

import streamlit as st

from src.firestore_store import FirestoreStore
from src.models import AppConfig
from src.slack_etl import run_pipeline

st.set_page_config(page_title="Slack Knowledge Export", layout="wide")
LOCAL_TZ = datetime.now().astimezone().tzinfo


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
    with st.spinner("Running export..."):
        try:
            result = run_pipeline(store, start_time=start_dt, end_time=end_dt)
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
            st.session_state.is_exporting = False


def render_logs(store: FirestoreStore) -> None:
    logs = store.fetch_logs(limit=50)
    if not logs:
        st.info("No logs recorded yet.")
        return
    st.dataframe(logs, use_container_width=True)


def build_download_bundle(output_dir: str) -> Optional[Tuple[bytes, str]]:
    base_path = Path(output_dir)
    if not base_path.exists():
        return None
    files = [path for path in base_path.rglob("*") if path.is_file()]
    if not files:
        return None
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            archive.write(file_path, arcname=str(file_path.relative_to(base_path)))
    buffer.seek(0)
    filename = f"slack_export_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.zip"
    return buffer.getvalue(), filename


def main() -> None:
    store = get_store()
    config = load_config(store)

    st.title("Slack Knowledge Base Exporter")
    st.caption("Incrementally export Slack knowledge into clean Markdown entries.")

    with st.sidebar:
        st.header("Configuration")
        slack_token = st.text_input(
            "Slack Bot Token",
            value=config.slack_token,
            type="password",
        )
        gemini_key = st.text_input(
            "Gemini API Key",
            value=config.gemini_key,
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
        gemini_model = st.text_input(
            "Gemini Model",
            value=config.gemini_model,
        )

        if st.button("Save Configuration", type="primary"):
            new_config = AppConfig(
                slack_token=slack_token.strip(),
                gemini_key=gemini_key.strip(),
                channel_ids=parse_channels(channels_raw),
                knowledge_base_dir=knowledge_dir.strip() or "knowledge_base",
                gemini_model=gemini_model.strip() or config.gemini_model,
            )
            save_config(store, new_config)
            st.success("Configuration saved.")

    st.subheader("Incremental Export")
    if st.button(
        "Run Incremental Export",
        disabled=st.session_state.get("is_exporting", False),
    ):
        trigger_pipeline(store)

    st.subheader("Manual Export (Custom Range)")
    today = date.today()
    default_start = today
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=today)
    if st.button(
        "Run Manual Export",
        disabled=st.session_state.get("is_exporting", False),
    ):
        start_dt = combine_date(start_date, end_of_day=False)
        end_dt = combine_date(end_date, end_of_day=True)
        trigger_pipeline(store, start_dt=start_dt, end_dt=end_dt)

    status = st.session_state.get("status_message")
    if status:
        level, message = status
        getattr(st, level)(message)
    if st.session_state.get("last_result"):
        last_result = st.session_state["last_result"]
        st.json(last_result)
        bundle = build_download_bundle(last_result["output_dir"])
        if bundle:
            content, filename = bundle
            st.download_button(
                "Download Exported Threads",
                data=content,
                file_name=filename,
                mime="application/zip",
            )
        else:
            st.caption("No exported files found to download yet.")
        st.metric("Threads Exported", last_result.get("threads_exported", 0))
        st.metric("LLM Summaries", last_result.get("summaries_generated", 0))
        st.metric("Transcript Fallbacks", last_result.get("summaries_fallback", 0))
        st.metric("Embedding Chunks", last_result.get("embedding_records", 0))
        summary_issues = last_result.get("summary_issues") or []
        if summary_issues:
            with st.expander("Summaries that fell back to raw transcript"):
                for issue in summary_issues:
                    st.write(f"- {issue}")

    st.subheader("Execution Logs")
    render_logs(store)


if __name__ == "__main__":
    main()
