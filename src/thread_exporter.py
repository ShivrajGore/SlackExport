from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .models import SlackThread


@dataclass
class RawThreadExporter:
    """Persists Slack thread transcripts verbatim as Markdown."""

    output_dir: Path

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, thread: SlackThread) -> Path:
        timestamp = datetime.fromtimestamp(float(thread.root_ts), tz=timezone.utc)
        slug = f"{thread.channel_id}-{timestamp.strftime('%Y%m%d-%H%M%S')}"
        safe_slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in slug)
        file_path = self.output_dir / f"{safe_slug}.md"
        content = self._render(thread, timestamp.isoformat())
        file_path.write_text(content, encoding="utf-8")
        return file_path

    def _render(self, thread: SlackThread, iso_ts: str) -> str:
        messages = []
        for message in thread.messages:
            ts = datetime.fromtimestamp(float(message.ts), tz=timezone.utc).isoformat()
            messages.append(f"[{ts}] {message.user}: {message.text}")
        joined = "\n".join(messages)
        return (
            f"# Raw Slack Thread\n"
            f"*Channel:* {thread.channel_id}\n"
            f"*Root timestamp:* {thread.root_ts} ({iso_ts})\n\n"
            f"## Conversation\n{joined}\n"
        )
