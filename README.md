# mcp-intern-adrig

This repository contains a small example project demonstrating an MCP-backed
assistant that exposes a backend API and a Vite React frontend, plus a
shopping-trends dataset used by the services.

**Quick overview**
- **Backend:** Starlette app entry at [src/cockpit_api.py](src/cockpit_api.py) which bridges to two MCP servers: [src/sales_server.py](src/sales_server.py) and [src/support_server.py](src/support_server.py).
- **Import script:** `scripts/import_shopping_trends_to_mongo.py` — loads dataset into MongoDB.
- **Frontend:** Vite + React app in `frontend/` (see [frontend/package.json](frontend/package.json)).
- **Dataset:** [dataset/shopping_trends.csv](dataset/shopping_trends.csv)

**Requirements**
- **Python:** 3.9+ (3.10+ recommended)
- **Node:** 16+ (for running the frontend dev server)
- **MongoDB:** A running MongoDB instance (default: `mongodb://localhost:27017`) or a hosted MongoDB URI

**Environment**
Create a `.env` file at the project root (optional — defaults are provided). Example:

```
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=intern_adrig
MONGODB_COLLECTION=shopping_trends
SHOPPING_TRENDS_CSV=dataset/shopping_trends.csv
# Schema-driven framework (default on): LLM only translates intent -> MCP JSON
SCHEMA_MODE=1
# LLM (Gemini) — required for full natural-language routing beyond schema resolver
# GEMINI_API_KEY=your_gemini_api_key
# GEMINI_MODEL=gemini-2.5-flash
# Skip LLM rewrite on raw MCP shopping results (framework formats tables itself)
# LLM_REPLY_MODE=rewrite
```

**Backend setup & run (Windows PowerShell)**
- **Create & activate venv:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- **Install Python dependencies (minimal):**

```powershell
pip install --upgrade pip
pip install python-dotenv pymongo starlette uvicorn langchain_mcp_adapters langchain-google-genai langchain_core mcp
```

- **Start MongoDB (Docker, if needed):**

```powershell
docker run -d --name mongo -p 27017:27017 -v mongo-data:/data/db mongo:6.0
```

- **Import CSV into MongoDB (optional — script also called by MCP tools):**

```powershell
python scripts/import_shopping_trends_to_mongo.py
```

- **Run the backend API:**

```powershell
uvicorn src.cockpit_api:app --reload --port 8000
```

- **Health check:**

```powershell
curl http://127.0.0.1:8000/api/health
```

**Frontend setup & run**
- From repository root:

```powershell
cd frontend
npm install
npm run dev
```

The Vite dev server runs on port `5173` by default. The backend allows CORS from the default Vite origin.

**How the system maps user queries to actions**
- The backend router in [src/cockpit_api.py](src/cockpit_api.py) decides whether to call MCP tools or return a direct answer.
- Common shopping-trends queries (top, count, color filters, review_rating summaries) are matched by a lightweight deterministic parser to avoid unnecessary LLM calls. This parser converts short user requests into MCP tool invocations (examples: `summarize_shopping_trends`, `query_shopping_trends`).
- For full natural-language routing the system uses Gemini when `GEMINI_API_KEY` is provided.

**Schema-Aware MCP Query Compiler (default)**
At startup the app samples MongoDB and builds a compact schema (columns, types, p20/p80). The LLM stage is **JSON-only**: natural language → MCP tool JSON. No chat prose. Results are formatted from MCP/MongoDB data in `ChatbotFramework`.

**Startup flow (schema discovery)**
- App starts.
- SchemaInspector samples up to 500 MongoDB docs.
- It builds a compact schema string (about 150-300 tokens, auto-discovered).
- LLMMCPBridge keeps this schema string and injects it into every LLM call.

**Request flow (example)**
- User: "top rated items"
- System prompt includes the compact schema:

```
Collection: shopping_trends
Columns:
	- review_rating: number [1.0-5.0] top>=4.2 bottom<=2.1
	- purchase_amount_usd: number [20-100] top>=84 bottom<=36
	- category: string values: Clothing, Footwear, Outerwear, Accessories
```

- LLM returns JSON only (about 50-120 tokens):

```json
{
	"tool": "query_shopping_trends",
	"params": {
		"filter": {"review_rating": {"$gte": 4.2}},
		"sort_by": "review_rating",
		"sort": "desc",
		"limit": 10
	}
}
```

- Bridge calls the MCP tool with those params.
- MCP queries MongoDB and returns rows.
- Backend returns the response to the user.

**Token budget (typical)**
```
System prompt  : ~60 tokens (instructions)
Schema hint    : ~150-300 tokens (from schema inspector)
User message   : ~10-30 tokens
LLM reply      : ~50-120 tokens (JSON only)
---------------------------------------------
Total per query: ~270-510 tokens
```

**Implementation outline (optional modules now included in src/)**
- [src/schema_inspector.py](src/schema_inspector.py):
	- Connects to MongoDB once at startup.
	- Samples the collection and infers:
		- column names
		- type (number | string | boolean)
		- numeric min, max, p20, p80
		- string values when cardinality is low
	- Exposes `build()` and `to_prompt()` for a compact schema string.

- [src/framework.py](src/framework.py):
	- Stage 1 strict JSON-only translation with validation and a one-time retry.
	- Stage 2 MCP dispatch with validated params.
	- Stage 3 deterministic table formatting (LLM fallback for unusual payloads).

- [src/llm_mcp_bridge.py](src/llm_mcp_bridge.py):
	- Sends (system prompt + schema + user message) to the LLM.
	- Enforces JSON-only responses.
	- Maps responses to MCP tool calls and returns results.

- Update [src/cockpit_api.py](src/cockpit_api.py) to:
	- Build schema once at startup.
	- Store the ChatbotFramework instance globally when `SCHEMA_MODE=1`.
	- On `/api/chat`, call `framework.chat(message)`.

**Example API requests**
- Chat request (JSON):

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
	-H "Content-Type: application/json" \
	-d '{"message":"top 5 categories"}'
```

- Import dataset via MCP tool (calls the import tool directly):

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
	-H "Content-Type: application/json" \
	-d '{"message":"import shopping trends"}'
```

**Developer notes & next improvements**
- The repository includes a local LLM fallback for offline development. For production or better results set `GEMINI_API_KEY`.
- The deterministic parser currently supports:
  - `top`, `most`, `summarize` -> `summarize_shopping_trends`
  - `how many`, `count` -> `query_shopping_trends` with `count_only`
  - color-based queries (e.g., "how many white items")
  - equality filters (e.g., "color white", "season winter")
  - `top rated` -> summarize on `review_rating`
- Possible enhancements: numeric range parsing (>, <, >=), $in lists, fuzzy matching for synonyms, schema-driven low-token mode, and unit tests for the parser.

**Key files**
- Backend entry: [src/cockpit_api.py](src/cockpit_api.py)
- MCP servers: [src/sales_server.py](src/sales_server.py), [src/support_server.py](src/support_server.py)
- Import script: [scripts/import_shopping_trends_to_mongo.py](scripts/import_shopping_trends_to_mongo.py)
- Frontend: [frontend/package.json](frontend/package.json)
- Dataset: [dataset/shopping_trends.csv](dataset/shopping_trends.csv)

If you'd like, I can:
- Add `requirements.txt` / `requirements-dev.txt` with pinned versions.
- Add parser unit tests under `tests/` and run them.
- Implement range parsing (e.g., `purchase_amount_usd > 50`) next.

---

Created/updated by the development assistant — tell me which follow-up you'd like.
