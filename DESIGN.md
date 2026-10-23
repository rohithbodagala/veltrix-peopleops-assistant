# Design

The LLM does two narrow jobs: a **router** classifies the message and extracts details (it never sees corpus text), and a **composer** writes document-grounded answers from pre-filtered, labelled excerpts (it can't call tools). Everything that touches permissions, data or rules is deterministic code. So a document can never trigger a tool, routes are reproducible, and personal data has few paths to the LLM.

## 1. Tables and the scanned document

Each format is parsed into ordered `Heading / Paragraph / Table` blocks (`app/ingestion/parsers/`).

- **PDF (PyMuPDF):** spans are grouped into lines, and headings are found from font size and weight. `find_tables()` locates the tables, but its own cell text dropped spaces in this corpus ("Lessthan2years", and "1.5" came out as "15\n."). So each cell is rebuilt from the spans inside the cell's box. Running headers and footers are dropped (small text in the margin band, or text repeated in that band across pages). Ligatures are NFKC-normalised.
- **DOCX:** the parser walks the document body in order, so a table stays under its heading. Styles are read from `w:pStyle`, because `paragraph.style` is `None` in these files.
- **HTML / Markdown:** headings, list items and tables are kept. Navigation, footer and style are dropped.
- **Tables** become one line per row with the header repeated (`Length of service: 2 years to less than 5 years | Days earned per month: 1.75`), so a row is still readable when it's retrieved on its own. Metadata boxes (Document ID, Version, Owner…) are removed generically; the manifest is the source of truth.
- **Scanned pages:** a PDF page with fewer than 40 characters of text and an image is rendered at 300 dpi and OCR'd with tesseract (installed in the image). Signature noise is filtered, and lines broken by the OCR layout are re-joined. A text-less DOCX or an image file goes through the same path.
- **Holidays** are extracted at ingestion from tables in `calendar` documents. The date column is found by parsing; the name column is the longest non-weekday text. A table or section labelled *restricted/optional* gives days the office stays open, which never reduce working days. Holidays are stored in memory and in `assistant_holidays`.

## 2. Chunking and retrieval

- **Chunks:** one chunk per section (heading plus body), split at paragraph boundaries above 450 tokens. Locator = the section heading, the title for text before the first section, or `p. N` for documents without headings. Sections are small in this corpus (20–300 tokens), so a chunk is usually exactly one citable rule.
- **Search:** BM25 over each router query, plus cosine similarity from a local CPU model (`bge-small-en-v1.5`, baked into the image), fused with reciprocal-rank fusion, with a relevance floor. Without the model it falls back to BM25 only.
- **Conflict expansion:** the top hits are used as queries to pull in passages from *other* documents that state the same rule. Without it, "doctor's note" found only the outdated FAQ and not the policy that overrides it, and precedence can't be applied to a passage that isn't there.
- **Why this design:** hybrid search covers both exact terms (leave types, place names) and paraphrases ("relative died" → bereavement), all locally, with no external store.

## 3. Section 5 rules

1. **Eligibility** (`retrieval/eligibility.py::is_eligible`) runs *before* scoring: role ≥ access level, region or GLOBAL, `effective_date ≤ reference date`.
2. **Reference date** is today, unless the router reports a past period. Then it's the last day of that period ("in 2025" → 2025-12-31), capped at today.
3. **Precedence** (`retrieval/precedence.py`): key = (authority, scope, effective date). FAQ and wiki are low authority; every other type, including unknown ones, is high. Regional beats GLOBAL. Then recency.
4. **Packing** (`retrieval/packer.py`): in-force excerpts S1…Sn are ordered and labelled with a dense **rank**; equal keys share a rank, which is a tie. Upcoming documents (readable, in the requester's region, not yet effective; only when the reference date is today) become U-excerpts with a computed `would_outrank` list of S labels.
5. **The composer prompt** applies the rules from these labels: better rank wins; same rank with different values = unresolved conflict (give both values, cite both); mention a U-excerpt only if its `would_outrank` includes the excerpt used. The model only has to decide whether two passages state the same rule; all ordering is computed in code.
6. **Citations** are mapped from the returned labels to `(doc_id, locator)`. Unknown labels are dropped. No valid citation means `no_answer`.

## 4. Where access control is enforced

| What | Where |
|---|---|
| Identity and role from the DB; session owner (401/403) | `app/main.py::chat`, `orchestration/sessions.py::SessionStore.get_or_create` |
| Documents filtered before ranking | `retrieval/eligibility.py::is_eligible`, used in `orchestration/composer.py::retrieve` and passed to `RetrievalIndex.search(allowed_doc_ids)` |
| Last line of defence before any LLM call | `llm/client.py::LLMClient.complete_json` refuses blocks outside the allow-list or over 3,000 tokens, and any system prompt that isn't one of the static prompts |
| Leave data, inside the tools | `tools/authz.py::can_view_employee_data / can_use_team_leave`, called by `tools/leave_tools.py::_target`, `get_team_leave`; writes always use `ctx.requester` |
| Pre-check so a forbidden request is answered `refuse` | `orchestration/targets.py::resolve_target` searches only the employees the requester may see, so an existing and a missing employee get the same fixed message |
| Citations re-checked against readable documents | `orchestrator.py::_finalize` |

## 5. Personal data

- **Repository queries never select `email` or `phone`**, so those fields never reach a prompt or an answer.
- **Names** reach the LLM only through tool results the requester may see. Earlier conversation turns go to the router with names replaced by employee ids. Tool-only answers are rendered from templates and never pass through an LLM.
- **Every answer and every tool-call argument** goes through `pii.scrub_answer`: emails are masked as `x***@domain` and phone numbers removed.
- **Logs** are structured events (ids, routes, tool statuses, `corpus_tokens`), never message text. A formatter on every handler, including uvicorn's, masks emails, removes phones, and replaces every employee name loaded from the DB with its id.

## 6. Failure modes found

**Fixed during development:**
- PDF table cells losing spaces and decimal points.
- `paragraph.style = None` in the DOCX files.
- Ligatures in PDF text.
- Footers leaking into chunks and merging paragraphs across pages.
- India's restricted holidays counted as days off.
- OCR signature noise.
- The outdated FAQ retrieved without the overriding policy (conflict expansion).
- "tomorrow" split on "to" by the date-range parser.
- `422` errors from FastAPI mapped to the required `400`.
- **Cold start in Docker** took 214 s and 2.3 GB of RAM. `find_tables` cost 1–2 s per page, so it now runs only on pages with vector lines. The embedding batch was cut from 256 to 16 (default padding allocated about 2.4 GB). Result: 55 s and 0.6 GB.
- **A process pool for parsing stalled** inside a 4 GB Docker VM, so it was removed.

**Fixed after an independent review**, each with a regression test:
- A proposal survived a rejected modification and could still run.
- "Yes, cancel it" and "Please go ahead" weren't recognised as confirmations. A word-list check now runs first, with the router's `confirms_pending` as a fallback.
- `future_balance` ignored the target employee.
- "yesterday" and "last Friday" were rejected instead of asked about.
- ISO dates in holiday tables were parsed day-first. The row's weekday now decides.
- The injection pre-filter refused "can my manager override the rules…?".
- Names were sent to the composer in tool results.
- Long messages returned 400.

**Open:**
- Live LLM behaviour is unverified. No key was available, so prompts were checked only through a scripted fake and the SDK signatures; `make smoke` is the first thing to run.
- In BM25-only mode, conflict expansion is weaker.
- Whether two passages state "the same rule" is still the LLM's judgement.

## 7. At 10,000 documents

- **First to break: start-up.** Every boot re-parses, re-OCRs and re-embeds everything: about 55 s for 30 documents in Docker. At 10k documents, `/health` would miss the 10-minute limit. Fix: incremental ingestion keyed by the file hash already stored in `assistant_documents`, a persistent vector index (FAISS, pgvector or Qdrant), and OCR run in a parallel worker pool off the request path.
- **Next: scoring.** BM25 and cosine are computed over every chunk in Python. Fix: an inverted and vector index with role, region and date filters pushed down into the query.
- **Then: precedence inside 3,000 tokens.** Many versions of one policy compete for the budget. Fix: group documents into families (same policy, different versions or regions) and keep only the strongest eligible version of each before packing.
