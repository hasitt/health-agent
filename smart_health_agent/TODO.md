# Smart Health Agent — Kanban Board

**Vision:** an AI-powered personal health agent that integrates wearable, dietary, and subjective data, finds meaningful correlations, and delivers personalized, empathetic, actionable coaching through a conversational interface. Long-term target: the triathlon market — AI coaching that auto-adapts to the athlete's metrics.

**Principles:** factual display of raw data (LLM does interpretation) · highly personalized · privacy-first.

> Board columns below are ordered Backlog → Next Up → In Progress → Done.
> The **Done** column is history — skip it unless you specifically need to know
> what's already built (see CLAUDE.md).

---

## 🗓️ Backlog

- **Physiological thresholds** — fetch/store lactate threshold HR, FTP, and HR zones from Garmin; implement `PhysiologicalBand` in `workout_analyzer.py` (the pluggable seam already exists). Unlocks time-in-zone ("your easy run wasn't easy") and threshold-relative workout narratives.
- **Tier-2 workout stats** — lagged cross-correlation (HR responds to effort with delay), CUSUM (slow drift), MAD robust bands. Refinements to the analysis engine.
- **Annotated workout chart** — Plotly time-series of a session with excursions/phases marked; a `plot_workout` tool. (No activity-sample plotting exists yet.)
- **Enhanced visualization tab** — dedicated Graphs tab with time-series (stress, sleep, RHR, steps, body battery, weekly mood/energy) and correlation scatter/bar charts. Candidate correlations: carb timing vs deep sleep, steps vs next-day sleep score, evening strength vs next-day RHR, late caffeine vs REM%, Mg vs sleep latency, meditation vs HRV, omega-3 vs HRV, Na:K vs SpO2, fiber vs body-battery recharge, late dinner vs deep sleep, alcohol vs respiration, evening yoga vs morning stress, lunch sat-fat vs afternoon battery slump, B6 vs wake episodes.
- **Comprehensive data sources** — other wearables, lab results (blood/hormones/nutrients), genetic reports, weather API (heat/humidity correlations).
- **Proactive nudges** — LLM offers unprompted insights from real-time patterns ("HRV low today, consider recovery").
- **Voice interaction** — voice-to-text and text-to-speech for hands-free logging/queries.
- **Cloud + mobile app** — migrate backend to cloud, native iOS/Android, auth, push notifications. The path to the triathlon-market product.
- **Cronometer auto-sync** — replace manual CSV upload with the Cronometer export API flow (login → nonce → /export). See earlier research.
- **Port micronutrient parsing** from the sibling `unified-health-mcp` repo (or retire that repo).

## 📋 Next Up (To Do)

- **Quick Subjective Check-In (emoji tap-through)** — _requested 2026-08-08._ See detailed card below. Highest-value UX item: revives subjective tracking (dead since Aug 2025), which also feeds the mood/energy correlations that currently return "no data."
- **Validate workout analysis on external datasets** — import public run/ride `.fit`/CSV data (e.g. rikluost/athlete_hr_predict ~50 runs, GoldenCheetah OpenData) via a small `analyze_samples_df` entry point; confirm decoupling / change-points / HR-recovery generalize beyond Stan's own data. Cheap de-risking before building further.
- **Grade-adjusted pace** — normalize speed by gradient (from elevation) so a slowdown *uphill* isn't flagged as fatigue. Fixes a known false-positive class in the workout analyzer.
- **Fix `distance_m` parser** — Garmin sends cumulative distance as `sumDistance`, parser expects `directDistance` (`data_fetcher.py:481-492`); distance is 0% populated. Unlocks distance-based splits and pace-from-distance.

## 🚧 In Progress

- _(nothing active — post-run coach just shipped)_

## ✅ Done

- Garmin data infrastructure: MCP server + `garmin_mcp_adapter`, incremental sync, resumable historical backfill (12+ months, ~948k activity samples), body-battery from device.
- Cronometer manual CSV import (food/recipe entries, nutrients, caffeine, alcohol).
- Subjective wellbeing table + Daily Mood Tracker UI (`submit_mood_entry`).
- Trend analysis: stress consistency, steps↔sleep, activity↔RHR, caffeine/alcohol↔stress/mood; missing-metric guard.
- LLM: upgraded to qwen3.5:9b, Ollama capability-based tool detection, tool-calling Health Detective agent.
- Gradio tabbed UI (dashboard, mood tracker), verified end-to-end.
- Workout analysis engine (`workout_analyzer.py`): per-metric SD-band excursions, windowed context, cross-metric correlation.
- Tier-1 workout analyzers (`workout_advanced.py`): change-point segmentation, aerobic decoupling, HR recovery, rolling-baseline.
- **Post-run coach**: `analyze_workout` tool wired into the chat agent + coaching prompt — "analyze my last run" gives a live verdict/what-happened/what-to-work-on read.
- Test suite green (96/96), credentials scrubbed, legacy `garmin_utils` retired.

---

## Card: Quick Subjective Check-In (emoji tap-through)

**Goal:** make logging subjective wellbeing effortless so it actually happens. The key is prompting at the *right moments* with near-zero friction — a couple of taps, never a form.

**Interaction:** one question at a time, each answered by tapping one of three large smileys 🙁 / 😐 / 🙂. Tapping auto-advances to the next. A "skip" and "done for today" are always available.

**Prompt at relevant times (the crux — not just on open):**
- **Morning** (on first open of the day): mood, energy, sleep quality.
- **Post-workout** (when a new activity syncs): "how did that session feel?" — RPE / mood — captured while it's fresh, and pairs directly with the workout analysis.
- **Evening** (wind-down): stress, focus, motivation.
- Only surface a context if it has no entry yet for that window today; keep each dismissable.

**Implementation notes:**
- Reuse the existing write path: `submit_mood_entry(mood, energy, stress, sleep_quality, focus, motivation, ...)` → `db.upsert_subjective_wellbeing` (`smart_health_ollama.py:507`). No new table — `subjective_wellbeing` already stores these 1–10 (CHECK 1..10).
- Map 3 smileys onto the 1–10 scale (🙁=2, 😐=5, 🙂=8). A 5-face variant can come later; start with 3 for speed.
- **Invert valence per metric:** for stress, 🙂 must mean *low* stress, so a happy face is always the "good" answer.
- Once flowing, revives the mood/energy correlations that currently return "no data recorded" for recent windows.
