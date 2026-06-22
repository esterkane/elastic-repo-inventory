# CLAUDE.md — Elastic Repo Inventory

Local-first release-intelligence app for Elasticsearch documentation. Syncs selected
Elastic repos, indexes Markdown into deterministic, provenance-carrying chunks, and
serves topic- and version-aware briefings via hybrid (lexical + vector) retrieval.
This is an active prototype, not production software.

## Run / test commands

All commands are PowerShell-friendly and match CI (`.github/workflows/ci.yml`).

Backend tests (from repo root):
```powershell
python -m pip install -e ".[dev]"
python -m pytest -p no:cacheprovider
```

Frontend tests + build (from `frontend/`):
```powershell
cd frontend
npm install
npm test -- --run        # vitest run
npm run build            # tsc -b && vite build
```

Compose validation + integration smoke (as CI runs it):
```powershell
docker compose config --quiet
docker compose build api frontend
docker compose up -d postgres qdrant
docker compose up -d --no-deps api
# health: GET http://localhost:8000/api/v1/health
```

Full local stack:
```powershell
docker compose up -d --build
# Frontend http://localhost:5173 · API http://localhost:8000/api/v1/health
# Qdrant http://localhost:6333 · Prometheus http://localhost:9090
```

Inventory CLI (writes `artifacts/repo-manifest.{json,md}`, clones into `sources/`):
```powershell
python tools/repo_inventory.py
```

MCP server (read-only agent tools over the retrieval core; reuses the API's env vars):
```powershell
python -m backend.app.mcp.server          # stdio (default)
$env:MCP_TRANSPORT="http"; python -m backend.app.mcp.server   # streamable-HTTP
```
Tools: `hybrid_search`, `get_chunk`, `rerank`, `list_sources`. See `docs/mcp.md`.

Deterministic evaluation (manual `eval.yml` workflow): runs `python -m pytest backend/tests`
and records NDCG@10 / MRR@10 / Recall@20 expectations to `artifacts/eval/summary.txt`.

Retrieval-quality evaluation via the shared **`relevance_eval`** skill (optional `eval`
extra, a git dependency). A thin adapter (`backend/app/eval/skill_adapter.py`) injects the
async hybrid `RetrievalService` into the skill's `search_fn(query, strategy) -> [chunk_id]`
contract; the runner gates Precision@k / MRR@k / nDCG@k against thresholds. One pipeline =
one strategy (`hybrid`). Install/run from `backend/`:
```powershell
python -m pip install -e ".[eval]"
python -m app.eval.run_eval          # writes reports/retrieval_eval.{json,md}; non-zero on gate fail
```
A real run needs a live stack (Postgres + Qdrant + TEI); offline unit tests
(`backend/tests/test_eval_skill_integration.py`) cover the wiring with a fake service.

NOTE: There is no checked-in linter or type-checker config (no ruff/mypy/eslint setup,
no Makefile, no `config.example.yaml`). The quality gate is: backend pytest, frontend
build, and the Compose/API integration job — nothing else is enforced. Do not invent
lint/type commands.

## Architecture in 5 lines

1. `tools/repo_inventory.py` syncs Elastic repos into managed cache mirrors under `sources/` and writes deterministic manifests to `artifacts/`.
2. `backend/app/ingest/` parses Markdown, normalizes metadata, and chunks deterministically (`sha256(repo:path:anchor:chunk_index)`); only changed chunks (by content hash) are re-embedded.
3. Chunks land in PostgreSQL (pgvector, lexical full-text) and Qdrant (dense vectors via the TEI embedding service); `backend/app/retrieval/` fuses both with RRF, metadata boosts, source de-dup, and optional TEI rerank.
4. `backend/app/recommend/` + answer synthesis build a release-aware briefing (answer / what's new / why it matters / evidence) — deterministic synthesis, NOT an LLM generation chain.
5. `frontend/` (React 19 + Vite) renders the briefing UI; the FastAPI app (`backend/app/main.py`, routers in `backend/app/api/`) exposes `/api/v1/*`. `backend/app/mcp/` exposes the same retrieval core as **read-only** MCP tools (`hybrid_search`, `get_chunk`, `rerank`, `list_sources`) — see `docs/mcp.md`.
6. `backend/app/agents/` adds an **opt-in** (`AGENTS_ENABLED`, default false) research-briefing multi-agent supervisor (`planner -> retrieval + rerank -> writer -> verifier`) that **wraps, never replaces** the hybrid retrieval and deterministic synthesis. The provenance-enforcing `VerifierAgent` (reusing `relevance_eval.faithfulness_score`) rejects any briefing with a claim that does not trace to a retained `source_url`.
7. `backend/app/generation/` adds an **opt-in** (`GENERATION_ENABLED`, default false) grounded-generation answer path **and its faithfulness gate** — the generation analogue of the relevance gate. `answer()` retrieves, drafts grounded-only via the reused provider/`WriterAgent`, verifies via the reused `VerifierAgent`, and returns `{text, citations[]}` with a `source_url` per claim. `run_gen_eval.py` mirrors `app/eval/run_eval.py`: it gates **citation-accuracy / faithfulness / answer-correctness** against `thresholds.example.json` and **exits non-zero on regression**, fully offline. The headline: **generation is gated on faithfulness the same way search is gated on relevance.**

## Agents architecture & invariants

The agents package wraps existing functions — no parallel retrieval/synthesis:

- **PlannerAgent** decomposes the question (deterministic rules by default; LLM via the provider). **RetrievalAgent**/**RerankAgent** call `RetrievalService.retrieve` / `RerankerClient.rerank` through `agents/tools.py`. **WriterAgent** drafts via the provider; the `DeterministicProvider` reuses `answer_evidence` + `synthesize_answer_model`, so the off/test path *is* the existing deterministic synthesis.
- **Deterministic synthesis is wrapped, not replaced.** When `AGENTS_ENABLED` is off the deterministic `/answer` path stays the default and is byte-for-byte unchanged; no agent route/job is active.
- **Every claim carries a retained `source_url`.** A briefing is only valid if every claim traces to a chunk that was actually retrieved.
- **Verification gate (the safety mechanism).** `VerifierAgent` checks per claim: token support (via `relevance_eval.faithfulness_score`, the shared eval skill) AND that the claim's `source_url` is in the retained set. **The briefing fails verification if any claim does not trace**, and the supervisor returns it `verified=false` with the offending claims rather than silently emitting it.
- **Reliability contract honoured.** Retrieval `warnings` / `degraded` are propagated and carried forward; a degraded stage does not crash the run; the warnings mechanism is reused, not reinvented.
- **Flag-gated, default-off.** `AGENTS_ENABLED` is read via `os.getenv` (there is no `config.py`), defaulting to false.
- **Provider-swappable.** The LLM lives behind one `AgentLlmProvider` protocol: `DeterministicProvider` (offline/test default), `OllamaProvider` (documented local default, wired `OLLAMA_URL`/`OLLAMA_MODEL`). Tests never require a network.
- **Reuse the eval skill.** Citation-accuracy + answer-faithfulness are measured on the committed `backend/app/agents/fixtures/briefing_eval.json` fixture via `relevance_eval.faithfulness_score` — fully offline (`backend/app/agents/eval.py`).

## Generation architecture & invariants

`backend/app/generation/` is the optional grounded-generation answer path plus
its faithfulness gate. It **reuses** the agents' provider/writer/verifier — there
is no second provider interface, writer, or verifier.

- **Gated on faithfulness, like search on relevance.** This is the headline. `run_gen_eval.py` mirrors `app/eval/run_eval.py`: load fixture + thresholds → eval → `evaluate_thresholds` → JSON/MD via `to_json`/`to_markdown` → exit 0/1. It **exits non-zero on regression**. Offline (deterministic provider + canned chunks) so it runs in CI.
- **Grounded-only, `source_url` per claim.** `answer()` retrieves, drafts via the reused `WriterAgent` (provider), and runs the reused `VerifierAgent`; only claims that *trace* (token-supported AND citing a retained `source_url`) become citations. Untraced claims are dropped and surfaced in `dropped_claims`. The return shape is `{text, citations[]}` with a `source_url` on every citation.
- **`GENERATION_ENABLED` default off (opt-in).** Read via `os.getenv` (no `config.py`), distinct from `AGENTS_ENABLED`. When off, the default answer path is the existing deterministic `/answer` synthesis, byte-for-byte unchanged; nothing here runs (`backend/tests/test_generation.py` proves off-by-default).
- **Three metrics, reusing the eval skill.** citation-accuracy (from the reused `VerifierAgent`), faithfulness (`relevance_eval.faithfulness_score`; the offline default uses token support — true contradiction detection is the LLM/NLI extension behind the scorer interface), answer-correctness (token-overlap F1 vs a committed gold answer). The eval shapes these into the `relevance_eval` report schema so the shared gate/renderer are reused unchanged.
- **Catches the unfaithful answer.** A committed fixture injects an answer whose claim its cited source does not support → faithfulness < 1.0 and the gate FAILS when thresholds require full faithfulness; a fully-grounded fixture passes. A test asserts both.
- **Provider-swappable, no secrets.** `DeterministicProvider` (offline/test default) or the wired local `OllamaProvider` (`AGENTS_PROVIDER=ollama`, `OLLAMA_URL`/`OLLAMA_MODEL`). Tests never require a network; no secrets in git.

## Invariants I must never break

1. **Determinism of the pipeline.** Chunk IDs are `sha256(repo:path:anchor:chunk_index)`; discovery/manifest output is `sorted(...)`; retrieval ordering and eval expectations are deterministic. Do not introduce nondeterministic ordering, IDs, or set/dict iteration that leaks into output.
2. **Passing the quality gate.** All three CI jobs must stay green: backend `pytest`, frontend `npm run build`, and the Compose+API integration job (`docker compose config --quiet`, image build, `/api/v1/health`). Add a test under `backend/tests/` or `frontend/src/**` for new behavior.
3. **Provenance on every chunk.** Every transformed/chunked/embedded/reranked/summarized record MUST retain: deterministic chunk ID, `repo`, `path`, commit SHA, content hash, canonical GitHub `source_url`, heading path/anchor when available, `content_type`, and `license_family`. No answer or briefing may exist that cannot be traced back to direct source links.
4. **No secrets in git.** Compose uses env-var interpolation with non-secret dev defaults (e.g. `${POSTGRES_PASSWORD:-repo_inventory}`); no `.env` is committed and none is required to exist. Never hardcode real credentials or commit a populated `.env`.

Repo-specific invariants:
- **Hybrid retrieval, not single-mode.** Combine lexical (Postgres) + vector (Qdrant) with RRF. Reranking is optional (TEI rerank profile / `TEI_RERANK_URL`).
- **Graceful degradation (reliability contract).** If vector fails → return lexical with a warning; if lexical fails → return vector with a warning; if rerank fails/disabled → return fused results, mark rerank skipped in explain mode; if one repo's sync fails → index the rest and report the failed repo.
- **Active source set only.** Ingest `elastic/docs-content`, `elastic/elasticsearch-labs`, `elastic/labs-releases`. Deprecated `elastic/docs` and `elastic/docs-builder` stay out. Serverless content is not promoted unless the query asks for it.
- **MCP tools are thin and read-only.** `backend/app/mcp/` adapters must contain no business logic — they validate input, call existing retrieval functions, and shape output (matching the HTTP API). Never expose ingestion/admin as a tool without an `MCP_ALLOW_MUTATIONS` flag defaulting to `false`. Every tool returns a structured result or a structured error — never a raw exception/stack trace.

## Definition of done

- [ ] Backend `python -m pytest -p no:cacheprovider` passes; new backend modules have tests in `backend/tests/`.
- [ ] Frontend `npm test -- --run` and `npm run build` pass (TypeScript strict via `tsc -b`).
- [ ] `docker compose config --quiet` succeeds and the API `/api/v1/health` integration path still works.
- [ ] Type checks: N/A (no mypy/eslint config checked in) — rely on `tsc -b` for TS and `pytest` for Python.
- [ ] Provenance intact: every new/changed chunk record keeps full source metadata; no untraceable answer paths.
- [ ] README/docs updated when behavior, config vars, or the source-repo set change.
- [ ] No secrets added; Compose stays on env-var interpolation with non-secret defaults.
- [ ] If MCP tools changed: they stay thin (no business logic), read-only, and validated; `docs/mcp.md` + the README "Agent Access" section are updated; `backend/tests/test_mcp_tools.py` covers the change.
- [ ] If the agents package changed: deterministic synthesis stays wrapped (not replaced) and `/answer` is unchanged with `AGENTS_ENABLED` off; every briefing claim carries a retained `source_url`; `VerifierAgent` still rejects any non-tracing claim (test proves an unsupported claim is caught); the reliability contract is honoured; the provider stays swappable; `backend/app/agents/eval.py` reuses `relevance_eval.faithfulness_score`; `backend/tests/test_agents.py` covers the change and runs fully offline.
- [ ] If the generation package changed: `GENERATION_ENABLED` stays off by default and the deterministic `/answer` path is unchanged when off (test proves it); `answer()` returns `{text, citations[]}` with a `source_url` on every citation, grounded only in retrieved chunks; it **reuses** the provider/`WriterAgent`/`VerifierAgent` (no duplicate provider/verifier); the eval reports citation-accuracy + faithfulness + answer-correctness and `run_gen_eval.py` gates them and exits non-zero on regression; a committed fixture proves the gate catches an unfaithful answer; `backend/tests/test_generation.py` covers the change and runs fully offline; no secrets added.

## External services & config

Config is supplied entirely through environment variables in `docker-compose.yml`
(no `.env.example`, no YAML config file). Key services and vars:

- **PostgreSQL** (`pgvector/pgvector:pg16`) — chunk storage + lexical retrieval. `DATABASE_URL` (async asyncpg URL, built from `POSTGRES_USER/PASSWORD/DB`, all defaulting to `repo_inventory`).
- **Qdrant** — dense retrieval/upserts. `QDRANT_URL` (`http://qdrant:6333`), `QDRANT_COLLECTION` (`repo-docs`).
- **TEI embedding service** — `TEI_EMBED_URL` (`http://tei-embed/embed`), `TEI_EMBED_MODEL` (`BAAI/bge-small-en-v1.5`).
- **TEI reranker (optional)** — Compose `rerank` profile, `TEI_RERANK_URL`, `TEI_RERANK_MODEL` (`BAAI/bge-reranker-base`).
- **Ollama `llm` (optional, future)** — present but unused; current answer path is deterministic synthesis, not LLM generation.
- **Prometheus (optional)** — local metrics at `:9090`.
- Ingestion tuning: `INGEST_EMBED_BATCH_SIZE` (8), `INGEST_UPSERT_BATCH_SIZE` (64), `SOURCES_DIR` (`/app/sources`).

Secrets: none required for dev; Compose interpolates env vars with non-secret defaults.
`.gitignore` excludes `.venv/`/`venv/` only — there is no committed `.env`.
