# VaultMind Frontend — SIH26117

This is the UI for VaultMind, talking to the backend agent over a REST API.
The backend team (Vision X) runs the agent on a RunPod GPU pod — you don't
need a GPU, or even the backend running locally, to build most of the UI.

## 1. Local development

Prerequisites: Node.js.

```bash
npm install
```

Create a `.env.local` file in this folder (this file is git-ignored, so
it's safe to put the real pod URL in it) pointing at the live backend:

```
VITE_API_URL=https://<pod-id>-8000.proxy.runpod.net
```

Get the actual `<pod-id>` URL from whoever has the RunPod pod running —
it's printed in the backend's startup logs, and stays stable across
restarts (it only changes if the pod itself is deleted and recreated).

```bash
npm run dev
```

Runs on `localhost:5173` (Vite default), talking to the real remote
backend the whole time.

## 2. How to call the API

Use an env-based helper so the same code works in dev (remote pod) and
production (same-origin, see §3) without changes:

```js
const API_BASE = import.meta.env.VITE_API_URL || ""; // "" = same-origin
```

### `GET /health`
```json
{ "status": "ok", "lightweight_mode": false, "models": { "reasoning": "...", ... } }
```
Good for a "backend connected" indicator in the UI.

### `GET /network-status`
```json
{ "external_calls_since_start": 0, "log": [] }
```
Wire this into the "0 external calls" sovereignty badge — this is the
number that matters most for judges.

### `POST /chat-json` — text-only requests
```js
const res = await fetch(`${API_BASE}/chat-json`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "Summarize the P-104 incident and draft an approval note." }),
});
const data = await res.json();
```
Response:
```json
{
  "answer": "Final text answer...",
  "trace": [
    { "step": "router", "detail": "Classified as: document" },
    { "step": "agent", "detail": "Plans: document_search({...})" },
    { "step": "tool_call", "tool": "document_search", "detail": "..." },
    { "step": "agent", "detail": "Plans: file_write({...})" },
    { "step": "tool_call", "tool": "file_write", "detail": "Saved deliverable to ..." }
  ]
}
```
**`trace` is exactly what should drive the live agent-trace panel** —
render each entry as a step appears, or render the full list at once as
a first pass before building real streaming.

### `POST /chat` — multipart, for requests with an uploaded image
```js
const form = new FormData();
form.append("message", "What does this schematic show?");
form.append("image", fileInput.files[0]);

const res = await fetch(`${API_BASE}/chat`, { method: "POST", body: form });
const data = await res.json(); // same { answer, trace } shape
```

## 3. Production build — same-origin deployment (what we use for the demo)

For the actual demo, your build gets served **by the backend itself** —
no separate frontend URL, no CORS, one process, one link for judges.

```bash
npm run build
```

This produces a `dist/` folder. Hand that folder to whoever's running
the backend on RunPod — it gets renamed `frontend_dist` and placed next
to `vaultmind_backend.py`, which auto-detects and serves it at the same
origin as the API.

**Important:** because `VITE_API_URL` won't be set at build time for
production, `API_BASE` resolves to `""` automatically — meaning your
fetch calls become relative (`/chat-json` instead of a full URL), which
is exactly what same-origin deployment needs. You don't need to change
any code between dev and production, just don't hardcode the pod URL
anywhere outside the `.env.local` file.

## 4. Getting your code into the shared repo

```bash
git clone https://github.com/<your-username>/VaultMind-SIH26117.git
cd VaultMind-SIH26117/frontend
# copy your project's files in here (package.json, src/, index.html, etc.)
# do NOT copy node_modules or dist — see .gitignore
git add .
git commit -m "Add frontend"
git push origin main
```

If you already have your own frontend repo, the simplest path is often
to copy its contents into this `frontend/` folder (not `git clone` it
as a nested repo) so everything lives together for the SIH submission.
