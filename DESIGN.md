# Design

The LLM has two narrow jobs: the **router** classifies the message and never sees corpus text; the **composer** writes document-grounded answers from pre-filtered, labelled excerpts and has no tools. Permissions, data access and rules are deterministic code, so a document can never trigger an action.

## 1. Tables and the scanned document

- **Common shape.** Every format is parsed into ordered `Heading / Paragraph / Table` blocks (`app/ingestion/parsers/`).
- **PDF (PyMuPDF):**
  - Headings are found from font size and weight. Running headers and footers are dropped: small text in the margin band, or text repeated across pages. Ligatures are NFKC-normalised.
  - `find_tables()` locates tables, but its own cell text lost spaces and decimals in this corpus ("Lessthan2years"; "15\n." for 1.5). Each cell is rebuilt from the text spans inside its box.
  - Table detection runs only on pages that have ruling lines; it costs 1–2 s per page.
- **DOCX:** the parser walks the body in order, so a table stays under its heading. Styles are read from `w:pStyle`, because `paragraph.style` is `None` in these files.
- **HTML and Markdown:** headings, list items and tables are kept; navigation, footer and style are dropped.
- **Tables** become one line per row with the header repeated, for example `Length of service: 2 years to less than 5 years | Days earned per month: 1.75`, so a retrieved row is still self-describing. Metadata boxes are removed; the manifest is the source of truth.
- **Scanned pages:** a page with almost no text and an image is rendered at 300 dpi and OCR'd with tesseract. Signature noise is filtered and broken lines are re-joined. Image-only DOCX files and image files use the same path.
- **Holidays** come from tables in `calendar` documents, extracted at ingestion:
  - The date column is detected by parsing. ISO dates are read as ISO; for numeric dates, a weekday in the same row breaks ties.
  - Tables or sections labelled *restricted/optional* list days the office stays open, so they never reduce working days.

## 2. Chunking and retrieval

- **Chunks:** one per section (heading plus body), split above 450 tokens. The locator is the heading, the title for text before the first section, or `p. N` when the document has no headings. Sections here are 20–300 tokens, so a chunk is usually one citable rule.
- **Search:** BM25 over the router's queries, plus cosine similarity from a local CPU model (`bge-small-en-v1.5`, baked into the image), fused by reciprocal rank. If the model is missing, it falls back to BM25.
- **Conflict expansion:** the top hits are used as queries to pull in passages from *other* documents that state the same rule. Without it, "doctor's note" found only the outdated FAQ, so precedence couldn't be applied.
- **Why this design:** it covers exact terms and paraphrases ("relative died" → bereavement), all locally, with no external store.

## 3. Section 5 rules

1. **Eligibility** (`retrieval/eligibility.py::is_eligible`) runs *before* scoring: role vs access level, region or GLOBAL, and effective date on or before the reference date.
2. **Reference date** is today, or the last day of a past period the user asks about, capped at today.
3. **Precedence key** (`retrieval/precedence.py`) is (authority, scope, effective date). FAQ and wiki are low authority; every other type, including unknown ones, is high.
4. **Packing** (`retrieval/packer.py`): in-force excerpts S1…Sn are sorted and labelled with a dense **rank**. Equal keys share a rank, which marks an unresolved tie. When the reference date is today, upcoming documents become U-excerpts with a computed `would_outrank` list of S labels.
5. **The composer** follows the ranks. Same rank with different values means it gives both and cites both. It mentions a U-excerpt only when that excerpt outranks the source it used. Its only judgement is whether two passages state the same rule.
6. **Citations** are mapped from labels to `(doc_id, locator)`. Unknown labels are dropped; no valid citation means `no_answer`.

## 4. Where access control is enforced

| What | Where |
|---|---|
| Identity from the DB; session owner (401/403) | `main.py::chat`, `sessions.py::SessionStore.get_or_create` |
| Documents filtered before ranking | `eligibility.py::is_eligible`, used in `composer.py::retrieve`, then `RetrievalIndex.search(allowed_doc_ids)` |
| Last check before any LLM call | `llm/client.py::LLMClient.complete_json` refuses non-allowed documents, more than 3,000 corpus tokens, and non-static system prompts |
| Leave data, inside the tools | `tools/authz.py::can_view_employee_data / can_use_team_leave`, used by `leave_tools.py::_target` and `get_team_leave`; writes use `ctx.requester` only |
| Forbidden request → `refuse` | `targets.py::resolve_target` searches only visible employees, so "exists" and "doesn't exist" get the same message |
| Citations re-checked | `orchestrator.py::_finalize` |

## 5. Personal data

- **Never loaded:** repository queries never select `email` or `phone`.
- **Names:** they reach the LLM only in the user's own words. Earlier turns go to the router with names replaced by ids, and tool results reach the composer without names. Tool answers come from templates and never pass through an LLM.
- **Answers and tool arguments:** `pii.scrub_answer` masks emails as `x***@domain` and removes phone numbers.
- **Logs:** structured events only (ids, routes, tool statuses, `corpus_tokens`), never message text. A formatter on every handler, including uvicorn's, also masks emails, removes phones and replaces every employee name from the DB with its id.

## 6. Failure modes found

**Fixed:**
- Parsing: PDF cells losing spaces and decimals; DOCX styles reading as `None`; ligatures; footers leaking into chunks and merging pages; OCR signature noise.
- India's restricted holidays counted as days off.
- The outdated FAQ retrieved without the policy that overrides it.
- "tomorrow" split on "to" by the date parser.
- FastAPI returning `422` where the spec requires `400`.
- **Cold start in Docker** took 214 s and 2.3 GB, because of table detection on every page and an embedding batch of 256. It now takes 55 s and 0.6 GB. A parallel parsing pool stalled in a 4 GB VM and was removed.

**Fixed after an independent review**, each with a regression test:
- A proposal survived a rejected modification and could still run.
- "Yes, cancel it" wasn't recognised as a confirmation.
- `future_balance` ignored the target employee.
- "yesterday" was rejected instead of asked about.
- ISO holiday dates were parsed day-first.
- The injection pre-filter refused ordinary questions.
- Names were sent to the composer.
- Long messages returned HTTP 400.

**Open:**
- Live LLM behaviour is unverified: no key was available, so the system was tested with a scripted fake and checked against the SDK signatures. Run `make smoke` first.
- Conflict expansion is weaker without embeddings.
- Deciding that two passages state "the same rule" depends on the LLM.

## 7. At 10,000 documents

- **First to break: start-up.** Every boot re-parses, re-OCRs and re-embeds the corpus: 55 s for 30 documents in Docker. At 10k documents, `/health` would miss the 10-minute limit. Fix: incremental ingestion keyed by the file hash already stored in `assistant_documents`, a persistent vector index (FAISS, pgvector or Qdrant), and OCR in a separate worker queue.
- **Next: scoring.** It scans every chunk in Python. Fix: an inverted and vector index with role, region and date filters pushed down into the query.
- **Then: precedence inside 3,000 tokens.** Many versions of one policy compete for the budget. Fix: group documents into families and keep only the strongest eligible version of each before packing.
