# Smart Health Agent

Local AI health agent: syncs Garmin (+ Cronometer) data into `health_data.db`, and a Gradio app (`smart_health_ollama.py`, localhost:7861) with a qwen3.5:9b tool-calling agent answers questions and coaches over it.

## Layout
- `smart_health_ollama.py` — Gradio app + LangChain agent + system prompt.
- `agent_tools.py` — LLM tools (registered in `HEALTH_AGENT_TOOLS`).
- `garmin-mcp-server/` — packaged MCP server (auth, fetching, models); `garmin_mcp_adapter.py` bridges it to the DB.
- `database.py` — SQLite schema + accessors.
- `trend_analyzer.py` — daily-granularity correlation/trend analysis.
- `workout_analyzer.py` + `workout_advanced.py` — per-workout time-series analysis engine.
- `poc_workout_analysis.py` — standalone harness for the workout engine.

## Working notes
- Runs on system `python3` (pyenv env `smart-health-agent`); no venv. Ollama must be running (`qwen3.5:9b`).
- Restart the app to pick up code changes: `pkill -f smart_health_ollama; python3 smart_health_ollama.py`.
- Server tests: `cd garmin-mcp-server && python3 -m pytest tests/ --override-ini addopts=`.

## TODO.md
`TODO.md` is a kanban board (Backlog → Next Up → In Progress → Done). **Do not read the "✅ Done" column unless you specifically need to confirm what's already built** — it's history and just consumes context. Work items live in Backlog / Next Up / In Progress.
