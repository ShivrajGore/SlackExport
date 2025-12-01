from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class AppConfig:
    """Holds application configuration that is persisted in Firestore."""

    slack_token: str
    gemini_key: str
    channel_ids: List[str]
    openai_api_key: str = ""
    knowledge_base_dir: str = "knowledge_base"
    gemini_model: str = "gemini-1.5-pro-latest"
    openai_model: str = "gpt-4o-mini"

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        return cls(
            slack_token=data.get("slackToken", ""),
            gemini_key=data.get("geminiKey", ""),
            openai_api_key=data.get("openaiApiKey", ""),
            channel_ids=data.get("channelIds", []),
            knowledge_base_dir=data.get("knowledgeBaseDir", "knowledge_base"),
            gemini_model=data.get("geminiModel", "gemini-1.5-pro-latest"),
            openai_model=data.get("openaiModel", "gpt-4o-mini"),
        )

    def to_dict(self) -> dict:
        return {
            "slackToken": self.slack_token,
            "geminiKey": self.gemini_key,
            "openaiApiKey": self.openai_api_key,
            "channelIds": self.channel_ids,
            "knowledgeBaseDir": self.knowledge_base_dir,
            "geminiModel": self.gemini_model,
            "openaiModel": self.openai_model,
        }


@dataclass
class SlackMessage:
    user: str
    text: str
    ts: str


@dataclass
class SlackThread:
    channel_id: str
    root_ts: str
    messages: List[SlackMessage] = field(default_factory=list)

    def as_prompt_block(self) -> str:
        """Return formatted text that is friendly for LLM prompts."""
        formatted = []
        for message in self.messages:
            timestamp = datetime.fromtimestamp(float(message.ts)).isoformat()
            formatted.append(f"[{timestamp}] {message.user}: {message.text}")
        return "\n".join(formatted)


@dataclass
class KnowledgeEntry:
    issue_description: str
    resolution: str
    findings: str
    source_channel: str
    source_ts: str
    file_path: Optional[Path] = None
    conversation: str = ""
    summary_provider: str = "transcript"
