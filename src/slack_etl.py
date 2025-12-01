from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from dateutil import parser as date_parser

import google.generativeai as genai
import openai
from google.api_core import exceptions as google_exceptions
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .embedding_store import EmbeddingWriter
from .firestore_store import FirestoreStore
from .models import AppConfig, KnowledgeEntry, SlackMessage, SlackThread

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an experienced Knowledge Engineer and meticulous incident scribe. You receive raw Slack threads describing incidents and troubleshooting steps.
Read every message carefully—never skip commands, error logs, or decision points—and capture the chronological story accurately. The audience is other engineers who rely on you for a precise record.

Format your answer as a JSON object with the following keys:
  - issue_description: Two to four sentences summarising the core issue.
  - resolution_fix: Ordered or bulleted steps that resolved the issue. Include commands or configuration values where possible.
  - findings_lessons: Broader findings, why it happened, preventative guidance, or validations to perform next time.

Keep answers factual and avoid guessing—quote the original phrasing when clarity matters. If a section truly lacks detail, write "Not documented", but only after double-checking the entire thread."""


def _format_timestamp(value: float) -> str:
    return f"{value:.6f}"


PLACEHOLDER_RESPONSES = {
    "not documented",
    "not provided",
    "not specified",
    "n/a",
    "na",
    "unknown",
    "not available",
}


def _normalize_response_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        parts = [_normalize_response_field(item) for item in value]
        joined = "\n".join(part for part in parts if part)
        return joined.strip()
    return str(value).strip()


def _extract_sections(payload: dict) -> tuple[str, str, str, List[str]]:
    issue = _normalize_response_field(
        payload.get("issue_description") or payload.get("issue")
    )
    resolution = _normalize_response_field(
        payload.get("resolution_fix") or payload.get("resolution")
    )
    findings = _normalize_response_field(
        payload.get("findings_lessons") or payload.get("findings")
    )
    issue = "" if issue.lower() in PLACEHOLDER_RESPONSES else issue
    resolution = "" if resolution.lower() in PLACEHOLDER_RESPONSES else resolution
    findings = "" if findings.lower() in PLACEHOLDER_RESPONSES else findings
    missing_sections: List[str] = []
    if not issue:
        missing_sections.append("issue description")
    if not resolution:
        missing_sections.append("resolution/fix")
    if not findings:
        missing_sections.append("findings/lessons")
    return issue, resolution, findings, missing_sections


def _build_prompt(thread: SlackThread, missing_sections: Optional[List[str]] = None) -> str:
    reminder = ""
    if missing_sections:
        missing_text = ", ".join(missing_sections)
        reminder = (
            "\n\nPrevious response omitted required sections. "
            f"Ensure the following sections contain actionable content: {missing_text}. "
            "Summarize concisely even if the thread is sparse."
        )
    return (
        "Respond with ONLY valid minified JSON (no code fences, no commentary).\n"
        f"Slack thread from channel {thread.channel_id}:\n"
        f"{thread.as_prompt_block()}"
        f"{reminder}"
    )


class SlackExtractor:
    """Fetches Slack messages and threads incrementally."""

    def __init__(self, token: str) -> None:
        self.client = WebClient(token=token)

    def fetch_threads(
        self,
        channel_id: str,
        oldest: Optional[float] = None,
        latest: Optional[float] = None,
    ) -> List[SlackThread]:
        messages = self._fetch_channel_history(channel_id, oldest, latest)
        threads: List[SlackThread] = []
        seen_threads: set[str] = set()
        for message in messages:
            thread_ts = message.get("thread_ts") or message.get("ts")
            if not thread_ts or thread_ts in seen_threads:
                continue
            seen_threads.add(thread_ts)
            thread_messages = [self._to_slack_message(message)]
            reply_count = message.get("reply_count", 0)
            if reply_count:
                replies = self._fetch_replies(
                    channel_id, thread_ts, oldest=oldest, latest=latest
                )
                for reply in replies:
                    if reply.get("ts") == message.get("ts"):
                        continue
                    thread_messages.append(self._to_slack_message(reply))
            threads.append(
                SlackThread(
                    channel_id=channel_id, root_ts=thread_ts, messages=thread_messages
                )
            )
        return threads

    def _fetch_channel_history(
        self,
        channel_id: str,
        oldest: Optional[float],
        latest: Optional[float],
    ) -> List[dict]:
        cursor: Optional[str] = None
        messages: List[dict] = []
        while True:
            params = {"channel": channel_id, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            if oldest is not None:
                params["oldest"] = _format_timestamp(oldest)
            if latest is not None:
                params["latest"] = _format_timestamp(latest)
            try:
                response = self.client.conversations_history(**params)
            except SlackApiError as exc:
                raise RuntimeError(f"Slack history error: {exc}") from exc
            messages.extend(response.get("messages", []))
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        return messages

    def _fetch_replies(
        self,
        channel_id: str,
        thread_ts: str,
        oldest: Optional[float],
        latest: Optional[float],
    ) -> Iterable[dict]:
        cursor: Optional[str] = None
        while True:
            params = {
                "channel": channel_id,
                "ts": thread_ts,
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor
            # Do not pass oldest/latest to ensure we capture the full thread
            # even if replies occur outside the requested window.
            try:
                response = self.client.conversations_replies(**params)
            except SlackApiError as exc:
                raise RuntimeError(f"Slack replies error: {exc}") from exc
            for message in response.get("messages", []):
                yield message
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break

    @staticmethod
    def _to_slack_message(payload: dict) -> SlackMessage:
        return SlackMessage(
            user=payload.get("user")
            or payload.get("username")
            or payload.get("bot_profile", {}).get("name", "system"),
            text=payload.get("text", ""),
            ts=payload.get("ts", "0"),
        )


def _conversation_summary(thread: SlackThread) -> str:
    if not thread.messages:
        return "Conversation transcript unavailable."
    first = thread.messages[0].text.strip()
    return first or "Conversation transcript captured below."


def _transcript_fallback_entry(
    thread: SlackThread,
    issue: str = "",
    resolution: str = "",
    findings: str = "",
) -> KnowledgeEntry:
    issue_text = issue or _conversation_summary(thread)
    default_text = "See conversation transcript for details."
    resolution_text = resolution or default_text
    findings_text = findings or default_text
    return KnowledgeEntry(
        issue_description=issue_text,
        resolution=resolution_text,
        findings=findings_text,
        source_channel=thread.channel_id,
        source_ts=thread.root_ts,
        conversation=thread.as_prompt_block(),
    )


class GeminiTransformer:
    """Calls Gemini to transform a Slack thread into a structured knowledge entry."""

    def __init__(self, api_key: str, model_name: str, max_attempts: int = 2) -> None:
        if not api_key:
            raise ValueError("Gemini API key missing from configuration.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_attempts = max(1, max_attempts)

    def transform_thread(self, thread: SlackThread) -> KnowledgeEntry:
        pending_sections: List[str] = []
        last_error: Optional[str] = None
        last_payload: Optional[tuple[str, str, str]] = None
        for attempt in range(self.max_attempts):
            user_prompt = f"{SYSTEM_PROMPT}\n\n{_build_prompt(thread, pending_sections or None)}"
            try:
                response = self.model.generate_content(user_prompt)
            except google_exceptions.GoogleAPIError as exc:
                last_error = str(exc)
                continue

            text = self._extract_json_text(response)
            if not text:
                last_error = "Gemini returned an empty response."
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                last_error = "Gemini response could not be parsed as JSON."
                continue

            issue, resolution, findings, pending_sections = _extract_sections(payload)
            last_payload = (issue, resolution, findings)
            if pending_sections:
                last_error = (
                    "Gemini response missing required fields: "
                    f"{', '.join(pending_sections)}"
                )
                continue

            return KnowledgeEntry(
                issue_description=issue.strip(),
                resolution=resolution.strip(),
                findings=findings.strip(),
                source_channel=thread.channel_id,
                source_ts=thread.root_ts,
                summary_provider="gemini",
            )

        if last_payload:
            issue, resolution, findings = last_payload
            logger.warning(
                "Gemini summary incomplete for thread %s: %s. Falling back to transcript context.",
                thread.root_ts,
                ", ".join(pending_sections) if pending_sections else "unknown issue",
            )
            entry = _transcript_fallback_entry(
                thread,
                issue=issue.strip(),
                resolution=resolution.strip(),
                findings=findings.strip(),
            )
            entry.summary_provider = "gemini-transcript"
            return entry

        raise RuntimeError(last_error or "Gemini transformation failed.")


class ChatGPTTransformer:
    """Calls OpenAI Chat Completions to transform a Slack thread."""

    def __init__(self, api_key: str, model_name: str, max_attempts: int = 2) -> None:
        if not api_key:
            raise ValueError("OpenAI API key missing from configuration.")
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model_name
        self.max_attempts = max(1, max_attempts)

    def transform_thread(self, thread: SlackThread) -> KnowledgeEntry:
        pending_sections: List[str] = []
        last_error: Optional[str] = None
        last_payload: Optional[tuple[str, str, str]] = None
        for attempt in range(self.max_attempts):
            user_prompt = _build_prompt(thread, pending_sections or None)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0,
                )
            except Exception as exc:  # pylint: disable=broad-except
                last_error = str(exc)
                continue

            choice = response.choices[0].message
            content = choice.content
            if isinstance(content, list):
                text = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            else:
                text = content or ""
            text = text.strip()
            if not text:
                last_error = "OpenAI returned an empty response."
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                last_error = "OpenAI response could not be parsed as JSON."
                continue

            issue, resolution, findings, pending_sections = _extract_sections(payload)
            last_payload = (issue, resolution, findings)
            if pending_sections:
                last_error = (
                    "OpenAI response missing required fields: "
                    f"{', '.join(pending_sections)}"
                )
                continue

            return KnowledgeEntry(
                issue_description=issue.strip(),
                resolution=resolution.strip(),
                findings=findings.strip(),
                source_channel=thread.channel_id,
                source_ts=thread.root_ts,
                summary_provider="openai",
            )

        if last_payload:
            issue, resolution, findings = last_payload
            logger.warning(
                "OpenAI summary incomplete for thread %s: %s. Falling back to transcript context.",
                thread.root_ts,
                ", ".join(pending_sections) if pending_sections else "unknown issue",
            )
            entry = _transcript_fallback_entry(
                thread,
                issue=issue.strip(),
                resolution=resolution.strip(),
                findings=findings.strip(),
            )
            entry.summary_provider = "openai-transcript"
            return entry

        raise RuntimeError(last_error or "OpenAI transformation failed.")

    @staticmethod
    def _extract_json_text(response) -> str:
        """Return raw JSON text, stripping common markdown fences."""
        text = (response.text or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Remove opening fence
            lines = lines[1:]
            # Drop closing fence if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            return text[start : end + 1]
        return text


class KnowledgeBaseLoader:
    """Writes the transformed entry to the local knowledge base directory."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_entry(self, entry: KnowledgeEntry) -> Path:
        ts = datetime.fromtimestamp(float(entry.source_ts), tz=timezone.utc)
        slug = f"{entry.source_channel}-{ts.strftime('%Y%m%d-%H%M%S')}"
        safe_slug = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in slug
        )
        file_path = self.output_dir / f"{safe_slug}.md"
        content = self._render_markdown(entry)
        file_path.write_text(content, encoding="utf-8")
        entry.file_path = file_path
        return file_path

    @staticmethod
    def _render_markdown(entry: KnowledgeEntry) -> str:
        conversation_block = ""
        if entry.conversation:
            conversation_block = (
                "\n## Conversation Transcript\n```\n"
                f"{entry.conversation.strip()}\n"
                "```"
            )
        return (
            f"# Issue Description\n{entry.issue_description.strip() or 'See conversation transcript below.'}\n\n"
            f"## Resolution / Fix\n{entry.resolution.strip() or 'See conversation transcript below.'}\n\n"
            f"## Findings & Lessons Learned\n{entry.findings.strip() or 'See conversation transcript below.'}\n\n"
            f"*Source: channel {entry.source_channel} at {entry.source_ts}*"
            f"{conversation_block}"
        )


def _build_embedding_text(entry: KnowledgeEntry, fallback: bool) -> str:
    parts = [
        f"Issue: {entry.issue_description.strip()}",
        f"Resolution: {entry.resolution.strip()}",
        f"Findings: {entry.findings.strip()}",
    ]
    if fallback:
        parts.append("Summary derived from transcript fallback.")
    if entry.conversation:
        snippet_lines = entry.conversation.splitlines()
        snippet = "\n".join(snippet_lines[: min(len(snippet_lines), 8)])
        parts.append(f"Conversation excerpt:\n{snippet}")
    return "\n".join(part for part in parts if part).strip()


def _validate_config(config: AppConfig) -> None:
    missing = []
    if not config.slack_token:
        missing.append("Slack token")
    if not (config.gemini_key or config.openai_api_key):
        missing.append("LLM API key (OpenAI or Gemini)")
    if not config.channel_ids:
        missing.append("channel IDs")
    if missing:
        raise ValueError(f"Missing configuration: {', '.join(missing)}")


def _to_epoch(value: Optional[datetime]) -> Optional[float]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def run_pipeline(
    store: FirestoreStore,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    channel_filter: Optional[Sequence[str]] = None,
) -> dict:
    """
    Execute the ETL pipeline.

    Args:
        store: Firestore store instance.
        start_time: Optional override start datetime. If None, uses stored state.
        end_time: Optional override end datetime.
        channel_filter: Optional subset of channels to process.
    """

    config = store.fetch_config()
    _validate_config(config)

    oldest = _to_epoch(start_time)
    latest = _to_epoch(end_time)
    manual_range = start_time is not None or end_time is not None
    if oldest is None:
        saved = store.fetch_last_run_ts()
        oldest = saved if saved is not None else 0.0

    extractor = SlackExtractor(config.slack_token)
    transformers: List[tuple[str, object]] = []
    if config.openai_api_key:
        transformers.append(
            (
                "openai",
                ChatGPTTransformer(
                    config.openai_api_key, config.openai_model, max_attempts=2
                ),
            )
        )
    if config.gemini_key:
        transformers.append(
            (
                "gemini",
                GeminiTransformer(config.gemini_key, config.gemini_model, max_attempts=2),
            )
        )
    loader = KnowledgeBaseLoader(config.knowledge_base_dir)
    embeddings_path = (
        Path(config.knowledge_base_dir) / "embeddings" / "threads.jsonl"
    )
    embedding_writer = EmbeddingWriter(embeddings_path)

    channels = list(channel_filter) if channel_filter else config.channel_ids
    exported = 0
    summaries_generated = 0
    summaries_fallback = 0
    failures: List[str] = []
    summary_issues: List[str] = []
    channel_stats: dict[str, int] = {channel: 0 for channel in channels}
    embedding_records = 0
    provider_stats: dict[str, int] = {}

    for channel_id in channels:
        try:
            threads = extractor.fetch_threads(channel_id, oldest=oldest, latest=latest)
        except Exception as exc:  # pylint: disable=broad-except
            failure = f"Channel {channel_id} extraction failed: {exc}"
            logger.exception(failure)
            failures.append(failure)
            continue
        if not threads:
            logger.info("No new threads for channel %s", channel_id)
            continue
        for thread in threads:
            fallback_used = False
            entry: Optional[KnowledgeEntry] = None
            provider_used: Optional[str] = None
            provider_errors: List[str] = []
            for provider_name, transformer in transformers:
                try:
                    entry = transformer.transform_thread(thread)
                    provider_used = provider_name
                    break
                except Exception as exc:  # pylint: disable=broad-except
                    error_msg = (
                        f"Thread {thread.root_ts} {provider_name} summary failed: {exc}"
                    )
                    logger.warning(error_msg)
                    provider_errors.append(error_msg)
            if entry is None:
                warning = (
                    f"Thread {thread.root_ts} fell back to transcript summary after "
                    f"providers failed: {', '.join(name for name, _ in transformers)}"
                )
                logger.warning(warning)
                summary_issues.append(warning)
                summary_issues.extend(provider_errors)
                entry = _transcript_fallback_entry(thread)
                fallback_used = True
                summaries_fallback += 1
                provider_used = None
            else:
                summaries_generated += 1
                summary_issues.extend(provider_errors)
            entry.summary_provider = provider_used or "transcript"
            entry.conversation = entry.conversation or thread.as_prompt_block()
            loader.save_entry(entry)
            exported += 1
            provider_key = entry.summary_provider
            provider_stats[provider_key] = provider_stats.get(provider_key, 0) + 1
            channel_stats[channel_id] = channel_stats.get(channel_id, 0) + 1
            embedding_writer.append(
                {
                    "id": f"{entry.source_channel}-{entry.source_ts}",
                    "channel_id": entry.source_channel,
                    "root_ts": entry.source_ts,
                    "fallback": fallback_used,
                    "provider": entry.summary_provider,
                    "text": _build_embedding_text(entry, fallback=fallback_used),
                }
            )
            embedding_records += 1

    if not manual_range and not failures:
        store.update_last_run(time.time())

    status = "SUCCESS" if not failures else "FAILURE"
    details = (
        f"Channels: {', '.join(channels)} | "
        f"Threads exported: {exported} | "
        f"LLM summaries: {summaries_generated} | "
        f"Transcript fallbacks: {summaries_fallback} | "
        f"Embedding chunks: {embedding_records} | "
        f"Errors: {len(failures)}"
    )
    store.append_log(status=status, details=details)

    return {
        "channels_scanned": len(channels),
        "threads_exported": exported,
        "summaries_generated": summaries_generated,
        "summaries_fallback": summaries_fallback,
        "embedding_records": embedding_records,
        "channel_stats": channel_stats,
        "failures": failures,
        "summary_issues": summary_issues,
        "summary_providers": provider_stats,
        "manual_range": manual_range,
        "start_timestamp": oldest,
        "end_timestamp": latest,
        "output_dir": str(loader.output_dir.resolve()),
    }


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = date_parser.isoparse(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Slack ETL pipeline runner")
    parser.add_argument(
        "--start",
        help="ISO-8601 datetime to override start time (inclusive). Defaults to stored state.",
    )
    parser.add_argument(
        "--end",
        help="ISO-8601 datetime to override end time (exclusive).",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        help="Specific channel IDs to process. Defaults to all configured channels.",
    )
    args = parser.parse_args()

    store = FirestoreStore()
    start_time = _parse_datetime(args.start)
    end_time = _parse_datetime(args.end)

    result = run_pipeline(
        store=store,
        start_time=start_time,
        end_time=end_time,
        channel_filter=args.channels,
    )
    logger.info("Pipeline finished: %s", result)


if __name__ == "__main__":
    main()
