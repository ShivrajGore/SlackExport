from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

from slack_sdk import WebClient

from .embedding_store import EmbeddingWriter
from .firestore_store import FirestoreStore
from .models import AppConfig, KnowledgeEntry, SlackMessage, SlackThread
from .slack_etl import (
    ChatGPTTransformer,
    GeminiTransformer,
    KnowledgeBaseLoader,
    _build_embedding_text,
    _transcript_fallback_entry,
)

PERMALINK_PATTERN = re.compile(
    r"slack.com/archives/(?P<channel>[A-Z0-9]+)/p(?P<ts>\d{10})(?P<ms>\d{6})"
)


def parse_permalink(permalink: str) -> tuple[str, str]:
    match = PERMALINK_PATTERN.search(permalink)
    if not match:
        raise ValueError(
            "Invalid Slack permalink. Expected slack.com/archives/<channel>/p<timestamp>"
        )
    channel = match.group("channel")
    ts = f"{match.group('ts')}.{match.group('ms')}"
    return channel, ts


def _build_thread(channel_id: str, messages: List[dict]) -> SlackThread:
    slack_messages: List[SlackMessage] = []
    for payload in messages:
        slack_messages.append(
            SlackMessage(
                user=payload.get("user")
                or payload.get("username")
                or payload.get("bot_profile", {}).get("name", "system"),
                text=payload.get("text", ""),
                ts=payload.get("ts", "0"),
            )
        )
    slack_messages.sort(key=lambda message: float(message.ts))
    root_ts = slack_messages[0].ts if slack_messages else "0"
    return SlackThread(channel_id=channel_id, root_ts=root_ts, messages=slack_messages)


def select_transformers(config: AppConfig) -> List[tuple[str, object]]:
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
    if not transformers:
        raise ValueError("Provide at least one LLM API key (OpenAI or Gemini).")
    return transformers


def transform_thread(
    transformers: List[tuple[str, object]], thread: SlackThread
) -> KnowledgeEntry:
    provider_errors: List[str] = []
    for provider_name, transformer in transformers:
        try:
            entry = transformer.transform_thread(thread)
            entry.summary_provider = provider_name
            return entry
        except Exception as exc:  # pylint: disable=broad-except
            provider_errors.append(f"{provider_name} failed: {exc}")
    entry = _transcript_fallback_entry(thread)
    entry.summary_provider = "transcript"
    if provider_errors:
        entry.findings += "\n\n" + "\n".join(provider_errors)
    return entry


def export_single_thread(config: AppConfig, permalink: str) -> tuple[KnowledgeEntry, dict]:
    channel_id, root_ts = parse_permalink(permalink)
    client = WebClient(token=config.slack_token)
    cursor: Optional[str] = None
    messages: List[dict] = []
    while True:
        params = {"channel": channel_id, "ts": root_ts, "limit": 200}
        if cursor:
            params["cursor"] = cursor
        response = client.conversations_replies(**params)
        messages.extend(response.get("messages", []))
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    if not messages:
        raise RuntimeError("Thread not found or no messages returned.")
    thread = _build_thread(channel_id, messages)
    transformers = select_transformers(config)
    entry = transform_thread(transformers, thread)
    entry.conversation = thread.as_prompt_block()
    loader = KnowledgeBaseLoader(config.knowledge_base_dir)
    loader.save_entry(entry)
    fallback_used = entry.summary_provider.endswith("transcript")
    record = {
        "id": f"{entry.source_channel}-{entry.source_ts}",
        "channel_id": entry.source_channel,
        "root_ts": entry.source_ts,
        "fallback": fallback_used,
        "provider": entry.summary_provider,
        "text": _build_embedding_text(entry, fallback=fallback_used),
    }
    embeddings_path = (
        Path(config.knowledge_base_dir) / "embeddings" / "threads.jsonl"
    )
    writer = EmbeddingWriter(embeddings_path)
    writer.append(record)
    return entry, record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a single Slack thread via permalink."
    )
    parser.add_argument("permalink", help="Slack thread permalink")
    args = parser.parse_args()

    store = FirestoreStore()
    config = store.fetch_config()
    entry, _ = export_single_thread(config, args.permalink)
    print(f"Thread exported to {entry.file_path}")


if __name__ == "__main__":
    main()
