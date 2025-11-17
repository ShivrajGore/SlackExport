# Slack Knowledge Base Export Agent

Implements the ETL workflow described in `AGENT.md`. The agent extracts Slack threads incrementally, transforms them with Gemini into structured knowledge entries, and saves Markdown files that can feed a knowledge base or custom GPT.

## Project Layout

- `src/slack_etl.py` – pipeline entrypoint and CLI for scheduled runs.
- `src/firestore_store.py` – persistence layer for configuration, state, and logs in Firestore.
- `src/models.py` – strongly typed dataclasses shared across components.
- `streamlit_app.py` – Streamlit dashboard for configuration management, manual exports, and log visibility.
- `knowledge_base/` – destination folder for generated Markdown files.

## Requirements

1. Python 3.10+
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Firebase credentials exposed through `GOOGLE_APPLICATION_CREDENTIALS` or passed via `--credentials` to the CLI/Streamlit app.
4. Firestore collections matching the design in `AGENT.md` (`config/settings`, `state/last_run`, `logs/*`).

## Running the Pipeline

### CLI / Scheduled Runs

```bash
python -m src.slack_etl \
  --start 2024-05-01T00:00:00Z \
  --end 2024-05-02T00:00:00Z \
  --channels C01ABCD1234 C01EFGH5678
```

Omit `--start`/`--end` for incremental runs that use the timestamp stored in Firestore. Only on successful incremental runs does the script update `/state/last_run`.

### Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Features:

- Manage Slack token, Gemini API key, monitored channels, output directory, and model selection.
- Trigger incremental exports or manual exports constrained by custom date ranges.
- View Firestore-backed execution logs.

## Knowledge Base Output

Each exported thread yields a Markdown file with three sections:

1. **Issue Description**
2. **Resolution / Fix**
3. **Findings & Lessons Learned**

File names are derived from the channel ID and root timestamp, ensuring unique, easily traceable entries inside `knowledge_base/`.

