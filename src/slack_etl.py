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
from google.api_core import exceptions as google_exceptions
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from .firestore_store import FirestoreStore
from .models import AppConfig, KnowledgeEntry, SlackMessage, SlackThread

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an experienced Knowledge Engineer. You receive raw Slack threads describing incidents and troubleshooting steps.
Carefully read the thread and distill it into three clear, concise sections. The audience is other engineers who need a crisp reference in the knowledge base.

Format your answer as a JSON object with the following keys:
  - issue_description: Two to four sentences summarising the core issue.
  - resolution_fix: Ordered or bulleted steps that resolved the issue. Include commands or configuration values where possible.
  - findings_lessons: Broader findings, why it happened, preventative guidance, or validations to perform next time.

Keep answers factual and avoid guessing. If information is missing, write "Not documented" for that section."""


def _format_timestamp(value: float) -> str:
    return f"{value:.6f}"


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
            if oldest is not None:
                params["oldest"] = _format_timestamp(oldest)
            if latest is not None:
                params["latest"] = _format_timestamp(latest)
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


class GeminiTransformer:
    """Calls Gemini to transform a Slack thread into a structured knowledge entry."""

    def __init__(self, api_key: str, model_name: str) -> None:
        if not api_key:
            raise ValueError("Gemini API key missing from configuration.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def transform_thread(self, thread: SlackThread) -> KnowledgeEntry:
        user_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            "Respond with ONLY valid minified JSON (no code fences, no commentary).\n"
            f"Slack thread from channel {thread.channel_id}:\n"
            f"{thread.as_prompt_block()}"
        )
        try:
            response = self.model.generate_content(user_prompt)
        except google_exceptions.GoogleAPIError as exc:
            raise RuntimeError(f"Gemini transformation failed: {exc}") from exc

        text = self._extract_json_text(response)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError("Gemini response could not be parsed as JSON.")

        issue = payload.get("issue_description") or payload.get("issue") or ""
        resolution = payload.get("resolution_fix") or payload.get("resolution") or ""
        findings = payload.get("findings_lessons") or payload.get("findings") or ""
        if not any([issue, resolution, findings]):
            raise RuntimeError("Gemini response missing required fields.")

        return KnowledgeEntry(
            issue_description=issue.strip(),
            resolution=resolution.strip(),
            findings=findings.strip(),
            source_channel=thread.channel_id,
            source_ts=thread.root_ts,
        )

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
        return (
            f"# Issue Description\n{entry.issue_description.strip()}\n\n"
            f"## Resolution / Fix\n{entry.resolution.strip()}\n\n"
            f"## Findings & Lessons Learned\n{entry.findings.strip()}\n\n"
            f"*Source: channel {entry.source_channel} at {entry.source_ts}*"
        )


def _validate_config(config: AppConfig) -> None:
    missing = []
    if not config.slack_token:
        missing.append("Slack token")
    if not config.gemini_key:
        missing.append("Gemini API key")
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
    transformer = GeminiTransformer(config.gemini_key, config.gemini_model)
    loader = KnowledgeBaseLoader(config.knowledge_base_dir)

    channels = list(channel_filter) if channel_filter else config.channel_ids
    exported = 0
    failures: List[str] = []
    channel_stats: dict[str, int] = {channel: 0 for channel in channels}

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
            try:
                entry = transformer.transform_thread(thread)
                loader.save_entry(entry)
                exported += 1
                channel_stats[channel_id] = channel_stats.get(channel_id, 0) + 1
            except Exception as exc:  # pylint: disable=broad-except
                error_msg = f"Thread {thread.root_ts} failed: {exc}"
                logger.exception(error_msg)
                failures.append(error_msg)

    if not manual_range and not failures:
        store.update_last_run(time.time())

    status = "SUCCESS" if not failures else "FAILURE"
    details = (
        f"Channels: {', '.join(channels)} | "
        f"Threads exported: {exported} | "
        f"Errors: {len(failures)}"
    )
    store.append_log(status=status, details=details)

    return {
        "channels_scanned": len(channels),
        "threads_exported": exported,
        "channel_stats": channel_stats,
        "failures": failures,
        "manual_range": manual_range,
        "start_timestamp": oldest,
        "end_timestamp": latest,
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
