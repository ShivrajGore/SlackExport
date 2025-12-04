# Slack Knowledge Base Export Agent

Implements the ETL workflow described in `AGENT.md`. The agent extracts Slack threads incrementally, transforms them with Gemini into structured knowledge entries, and saves Markdown files that can feed a knowledge base or custom GPT.

## Project Layout

- `src/slack_etl.py` – pipeline entrypoint and CLI for scheduled runs.
- `src/firestore_store.py` – lightweight local persistence layer for configuration, state, and logs (stored in `.data/`).
- `src/models.py` – strongly typed dataclasses shared across components.
- `streamlit_app.py` – Streamlit dashboard for configuration management, manual exports, and log visibility.
- `knowledge_base/` – destination folder for generated Markdown files.

## Requirements

1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Write access to the project directory so the app can create the `.data/` folder (config/state/log storage) and `knowledge_base/` output files.

## Local Persistence

When you interact with the Streamlit dashboard or CLI, the shared state lives in `.data/`:

- `.data/config.json` – Slack token, Gemini key, channel IDs, knowledge base directory, Gemini model.
- `.data/state.json` – timestamp of the last successful incremental run.
- `.data/logs.json` – append-only record of pipeline executions displayed in the Streamlit log table.

These files are created automatically on first run. Add `.data/` to `.gitignore` if you do not want to commit secrets or run history.

## Running the Pipeline

### CLI / Scheduled Runs

```bash
python -m src.slack_etl \
  --start 2024-05-01T00:00:00Z \
  --end 2024-05-02T00:00:00Z \
  --channels C01ABCD1234 C01EFGH5678
```

Omit `--start`/`--end` for incremental runs that use the timestamp stored in `.data/state.json`. Only on successful incremental runs does the script update that timestamp.

### Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Features:

- Manage Slack token, Gemini API key, monitored channels, output directory, and model selection.
- (Optional) Provide an OpenAI API key/model; OpenAI runs first, Gemini acts as a secondary fallback.
- Trigger an **Original Thread Export** to dump raw Slack transcripts (no LLM calls) for long historical backfills.
- Trigger incremental exports or manual exports constrained by custom date ranges.
- View the locally persisted execution logs (`.data/logs.json`).

## Knowledge Base Output

Each exported thread yields a Markdown file with three sections:

1. **Issue Description**
2. **Resolution / Fix**
3. **Findings & Lessons Learned**

File names are derived from the channel ID and root timestamp, ensuring unique, easily traceable entries inside `knowledge_base/`.

Each Markdown file now embeds both the structured summary and the full conversation transcript. A parallel JSONL file (`knowledge_base/embeddings/threads.jsonl`) captures embedding-friendly text chunks (issue, resolution, findings, and an excerpt) for indexing in custom GPT or vector databases.

### Exporting a Single Thread by Permalink

When you only need one thread, run:

```bash
python -m src.slack_single_export "https://yourworkspace.slack.com/archives/C01ABCD/p1717023456000123"
```

The CLI parses the permalink, fetches the full conversation, and writes the Markdown/embedding entry using the same OpenAI→Gemini→transcript fallback chain.
