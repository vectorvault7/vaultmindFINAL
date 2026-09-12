"""
VaultMind Backend — SIH26117
=============================
A self-hosted, multi-model Agentic AI backend built with LangGraph.

Runs on a rented RunPod GPU pod. Exposes a FastAPI HTTP API — reachable
either through RunPod's built-in HTTP port proxy (no tunnel software,
no API key, zero outbound connections from the pod) or directly over
your local network, so a separately-hosted frontend can call it.

--------------------------------------------------------------------
HOW TO RUN ON RUNPOD
--------------------------------------------------------------------
1. Create a Pod: choose a PyTorch template (recent CUDA + torch
   preinstalled — don't reinstall torch yourself, see requirements.txt).
2. Recommended GPU: RTX 4090, 24GB VRAM — see README.md for why this
   is the affordable sweet spot for this model stack.
3. In the pod's network settings, expose HTTP port 8000. RunPod gives
   you a public URL automatically: https://<pod-id>-8000.proxy.runpod.net
   No ngrok, no auth token, no outbound tunnel from inside the pod.
4. SSH or use the web terminal, then:
       pip install -r requirements.txt
       python vaultmind_backend.py
5. Send your teammate the RunPod proxy URL from step 3.

See README.md for the full API contract and model-swap instructions.
--------------------------------------------------------------------
"""

import os
import gc
import io
import re
import sys
import json
import time
import base64
import socket
import ipaddress
import subprocess
from typing import TypedDict, List, Optional, Literal

import torch

# =====================================================================
# 1. CONFIG
# =====================================================================
# LIGHTWEIGHT_MODE=True uses smaller, well-established models — handy if
# you ever fall back to testing on a smaller/shared GPU. Defaults to
# False here since RunPod's rented GPU is the real target: the full
# production stack we spec'd for the actual SIH demo.
LIGHTWEIGHT_MODE = os.environ.get("VAULTMIND_LIGHTWEIGHT", "0") == "1"

if LIGHTWEIGHT_MODE:
    MODELS = {
        "reasoning": "Qwen/Qwen2.5-7B-Instruct",
        "coding": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "vision": "Qwen/Qwen2.5-VL-7B-Instruct",
        "ocr": "deepseek-ai/DeepSeek-OCR-2",
    }
else:
    # Full production stack. Qwen3.6-27B and Qwen3-Coder-30B-A3B are
    # each ~15-18GB at 4-bit; only one heavy model is kept resident at
    # a time (see ModelManager below), so 24GB VRAM (RTX 4090) is enough.
    MODELS = {
        "reasoning": "Qwen/Qwen3.6-27B",
        "coding": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "vision": "Qwen/Qwen2.5-VL-7B-Instruct",
        "ocr": "deepseek-ai/DeepSeek-OCR-2",
    }

EMBEDDING_MODEL = "BAAI/bge-m3"
MAX_AGENT_ITERATIONS = 6
KNOWLEDGE_BASE_DIR = "vaultmind_kb"
SCRATCH_DIR = "vaultmind_scratch"
os.makedirs(KNOWLEDGE_BASE_DIR, exist_ok=True)
os.makedirs(SCRATCH_DIR, exist_ok=True)

print(f"[VaultMind] LIGHTWEIGHT_MODE={LIGHTWEIGHT_MODE}")
print(f"[VaultMind] Model stack: {json.dumps(MODELS, indent=2)}")

# =====================================================================
# 2. NETWORK MONITOR
# =====================================================================
# Tracks outbound connection attempts made AFTER models are loaded, so
# the demo can show "0 external calls" during actual inference — model
# downloads happen before this counter starts and are expected to hit
# the network (they pull weights from Hugging Face). Loopback/private
# addresses (localhost, your own LAN, RunPod's internal proxy) are
# deliberately excluded — those are legitimate local traffic, not the
# external egress this is meant to prove doesn't happen.
def _is_external(address) -> bool:
    host = address[0] if isinstance(address, tuple) else str(address)
    try:
        ip = ipaddress.ip_address(host)
        return not (ip.is_loopback or ip.is_private or ip.is_link_local)
    except ValueError:
        return host not in ("localhost",)


class NetworkMonitor:
    def __init__(self):
        self.enabled = False
        self.call_log = []
        self._orig_connect = socket.socket.connect

    def start(self):
        self.enabled = True
        monitor = self

        def patched_connect(sock_self, address, *a, **kw):
            if monitor.enabled and _is_external(address):
                monitor.call_log.append({"address": str(address), "ts": time.time()})
            return monitor._orig_connect(sock_self, address, *a, **kw)

        socket.socket.connect = patched_connect
        print("[NetworkMonitor] Started — watching for outbound connections from this point on.")

    def status(self):
        return {"external_calls_since_start": len(self.call_log), "log": self.call_log[-20:]}


net_monitor = NetworkMonitor()

# =====================================================================
# 3. MODEL MANAGER  (lazy load / unload — one heavy model resident at a time)
# =====================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)


class ModelManager:
    """Loads exactly one text-generation model at a time to fit Kaggle's
    free GPU memory. Swapping models takes a few seconds; that's an
    acceptable tradeoff for dev/testing (on the RunPod demo box with
    more VRAM you can raise this to keep 2+ resident simultaneously)."""

    def __init__(self):
        self.current_key = None
        self.model = None
        self.tokenizer = None

    def get(self, key: Literal["reasoning", "coding"]):
        if self.current_key == key and self.model is not None:
            return self.model, self.tokenizer
        self._unload()
        model_id = MODELS[key]
        print(f"[ModelManager] Loading '{key}' -> {model_id} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )
        self.current_key = key
        print(f"[ModelManager] '{key}' ready.")
        return self.model, self.tokenizer

    def _unload(self):
        if self.model is not None:
            print(f"[ModelManager] Unloading '{self.current_key}' ...")
            del self.model
            self.model = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()

    def generate(self, key, messages, max_new_tokens=800):
        model, tok = self.get(key)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok([text], return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.4, do_sample=True)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)


model_manager = ModelManager()

# Vision/OCR model kept separate since it's loaded far less often and
# has a different call signature (image input).
_vision_model = None
_vision_processor = None


def _pdf_to_image(pdf_path: str) -> str:
    """Converts page 1 of a PDF to a PNG so the OCR model (which expects
    an image) can read it. Multi-page reports: extend this to loop pages
    and concatenate results if your demo scenario needs more than page 1."""
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
    try:
        pages = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
    except PDFInfoNotInstalledError:
        raise RuntimeError(
            "pdf2image needs the 'poppler-utils' system package, which isn't "
            "installed. Run this in a Kaggle cell first, then retry:\n"
            "  !apt-get update -qq && apt-get install -y -qq poppler-utils"
        )
    out_path = pdf_path.rsplit(".", 1)[0] + "_page1.png"
    pages[0].save(out_path)
    return out_path


def run_vision_ocr(image_path: str, prompt: str = "Free OCR. Extract all text and describe any diagrams or tables.") -> str:
    global _vision_model, _vision_processor
    from transformers import AutoProcessor

    if image_path.lower().endswith(".pdf"):
        image_path = _pdf_to_image(image_path)

    if _vision_model is None:
        print(f"[VisionOCR] Loading {MODELS['ocr']} ...")
        _vision_processor = AutoProcessor.from_pretrained(MODELS["ocr"], trust_remote_code=True)
        try:
            _vision_model = AutoModel.from_pretrained(
                MODELS["ocr"], trust_remote_code=True, use_safetensors=True,
                attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16,
            ).eval().cuda()
        except Exception:
            # T4/P100 (Turing/Pascal) don't support flash-attn-2 — fall back to eager.
            _vision_model = AutoModel.from_pretrained(
                MODELS["ocr"], trust_remote_code=True, use_safetensors=True,
                attn_implementation="eager", torch_dtype=torch.bfloat16,
            ).eval().cuda()
    result = _vision_model.infer(_vision_processor, prompt=prompt, image_file=image_path)
    return result if isinstance(result, str) else str(result)

# =====================================================================
# 4. LOCAL KNOWLEDGE BASE  (RAG over SOPs / manuals / correspondence)
# =====================================================================
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"[KnowledgeBase] Loading embedder {EMBEDDING_MODEL} ...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


SAMPLE_DOCS = {
    "sop_pump_maintenance.txt": (
        "SOP-104: Centrifugal Pump P-104 Maintenance Guidelines.\n"
        "Inspect seal integrity every 30 days. Vibration threshold: 4.5 mm/s RMS max.\n"
        "Prior incidents (2025-11-02): bearing overheat traced to lubrication interval breach.\n"
        "Root cause: lubrication cycle extended past OEM recommendation of 90 days.\n"
        "Corrective action: enforce 60-day lubrication cycle, log in CMMS."
    ),
    "incident_log_p104_2026.txt": (
        "Incident Report IR-2026-014: Pump P-104 unplanned shutdown, 2026-02-18.\n"
        "Symptom: high vibration alarm followed by bearing temperature spike.\n"
        "Investigation found lubrication log gap of 74 days prior to failure.\n"
        "Cross-reference SOP-104: exceeds the 60-day enforced cycle.\n"
        "Recommended approval: replace bearing assembly, restore lubrication cadence."
    ),
    "oisd_156_summary.txt": (
        "OISD-STD-156 (Summary): Fire and Safety requirements for refineries and\n"
        "oil terminals. Section 4.3 covers rotating equipment safety clearances.\n"
        "Section 7.1 covers mandatory incident root-cause documentation before\n"
        "equipment is returned to service."
    ),
}

for fname, content in SAMPLE_DOCS.items():
    path = os.path.join(KNOWLEDGE_BASE_DIR, fname)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(content)

_kb_index = None
_kb_chunks = []


def build_knowledge_base():
    global _kb_index, _kb_chunks
    embedder = get_embedder()
    _kb_chunks = []
    for fname in os.listdir(KNOWLEDGE_BASE_DIR):
        with open(os.path.join(KNOWLEDGE_BASE_DIR, fname)) as f:
            text = f.read()
        for para in [p.strip() for p in text.split("\n") if p.strip()]:
            _kb_chunks.append({"source": fname, "text": para})
    vectors = embedder.encode([c["text"] for c in _kb_chunks], normalize_embeddings=True)
    _kb_index = faiss.IndexFlatIP(vectors.shape[1])
    _kb_index.add(np.array(vectors, dtype="float32"))
    print(f"[KnowledgeBase] Indexed {len(_kb_chunks)} chunks from {len(os.listdir(KNOWLEDGE_BASE_DIR))} documents.")


def tool_document_search(query: str, k: int = 3) -> str:
    if _kb_index is None:
        build_knowledge_base()
    embedder = get_embedder()
    qvec = embedder.encode([query], normalize_embeddings=True)
    scores, idxs = _kb_index.search(np.array(qvec, dtype="float32"), k)
    hits = [_kb_chunks[i] for i in idxs[0] if i < len(_kb_chunks)]
    if not hits:
        return "No relevant documents found."
    return "\n".join(f"[{h['source']}] {h['text']}" for h in hits)

# =====================================================================
# 5. TOOLS
# =====================================================================
from docx import Document as DocxDocument
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches, Pt


def tool_file_read(filename: str, max_chars: int = 4000) -> str:
    """Reads a file the agent previously wrote (or one already sitting in
    the scratch/knowledge-base folders) back into context — the explicit
    'file read' half of the brief's 'file read and write' tool pair."""
    candidates = [
        filename,
        os.path.join(SCRATCH_DIR, filename),
        os.path.join(KNOWLEDGE_BASE_DIR, filename),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return f"File not found: {filename} (looked in scratch and knowledge base folders)."
    if path.endswith(".docx"):
        doc = DocxDocument(path)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif path.endswith(".xlsx"):
        wb = load_workbook_readonly(path)
        text = wb
    else:
        with open(path, errors="ignore") as f:
            text = f.read()
    return text[:max_chars]


def load_workbook_readonly(path: str) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True)
    lines = []
    for ws in wb.worksheets:
        lines.append(f"[sheet: {ws.title}]")
        for row in ws.iter_rows(values_only=True):
            lines.append(", ".join(str(c) for c in row if c is not None))
    return "\n".join(lines)


def tool_file_write(content: str, filename: str, fmt: str = "docx") -> str:
    """Writes a real deliverable file. fmt: 'docx' | 'xlsx' | 'pptx' | 'txt'.

    For 'xlsx', `content` should be rows separated by newlines and columns
    by commas (e.g. "Item, Qty, Cost\\nBearing, 2, 4500"). For 'pptx',
    `content` should be slides separated by a line of '---', with the
    first line of each slide treated as the title and the rest as body
    bullets.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", filename)
    path = os.path.join(SCRATCH_DIR, safe_name if safe_name.endswith(f".{fmt}") else f"{safe_name}.{fmt}")

    if fmt == "docx":
        doc = DocxDocument()
        for line in content.split("\n"):
            if line.strip():
                doc.add_paragraph(line)
        doc.save(path)

    elif fmt == "xlsx":
        wb = Workbook()
        ws = wb.active
        for row in content.strip().split("\n"):
            ws.append([cell.strip() for cell in row.split(",")])
        wb.save(path)

    elif fmt == "pptx":
        prs = Presentation()
        for slide_text in content.strip().split("---"):
            lines = [l for l in slide_text.strip().split("\n") if l.strip()]
            if not lines:
                continue
            slide = prs.slides.add_slide(prs.slide_layouts[1])
            slide.shapes.title.text = lines[0]
            if len(lines) > 1:
                body = slide.placeholders[1].text_frame
                body.text = lines[1]
                for line in lines[2:]:
                    p = body.add_paragraph()
                    p.text = line
        prs.save(path)

    else:
        with open(path, "w") as f:
            f.write(content)

    return f"Saved deliverable to {path}"


def tool_code_exec(code: str, timeout: int = 10) -> str:
    """Runs Python in a subprocess with a timeout and no network env vars.
    This is a basic sandbox suitable for a hackathon demo, not a hardened
    production sandbox — for MRPL's real deployment this would run in a
    locked-down container with no filesystem/network access beyond a
    scratch volume."""
    script_path = os.path.join(SCRATCH_DIR, f"_exec_{int(time.time()*1000)}.py")
    with open(script_path, "w") as f:
        f.write(code)
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "http_proxy": "127.0.0.1:0", "https_proxy": "127.0.0.1:0"},
        )
        output = result.stdout + ("\n[stderr]\n" + result.stderr if result.stderr else "")
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout}s."
    finally:
        os.remove(script_path)


TOOLS_DESCRIPTION = """You have access to these tools. Respond with a JSON object
and nothing else, in the form: {"tool": "<name>", "args": {...}} to call a tool,
or {"tool": "final_answer", "args": {"answer": "..."}} when the task is complete.

Available tools:
- document_search(query): search the local knowledge base (SOPs, manuals, incident logs).
- file_read(filename): read back a file already saved in this session (e.g. a deliverable you wrote earlier, or a knowledge-base doc) — supports .docx, .xlsx, and plain text.
- file_write(content, filename, fmt): write a deliverable file. fmt is "docx", "xlsx", "pptx", or "txt".
  - xlsx: content = rows separated by newlines, columns separated by commas.
  - pptx: content = slides separated by a line of "---"; first line of each slide is the title, remaining lines are bullets.
- code_exec(code): run Python code in a sandbox and return stdout/stderr.
- vision_ocr(image_path): extract text/understand an uploaded scanned document, PDF, or photo (PDFs are converted automatically).
- final_answer(answer): use this once you have everything needed to answer the user.
"""

# =====================================================================
# 6. ROUTER
# =====================================================================
def route_task(user_message: str, has_image: bool) -> str:
    if has_image:
        return "image"
    code_signals = ["code", "script", "function", "python", "bug", "debug", "sql", "algorithm"]
    if any(sig in user_message.lower() for sig in code_signals):
        return "code"
    return "document"

# =====================================================================
# 7. LANGGRAPH AGENT LOOP
# =====================================================================
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    user_message: str
    image_path: Optional[str]
    task_type: str
    history: List[dict]     # [{"action": ..., "result": ...}, ...]
    trace: List[dict]       # streamed to the frontend
    final_answer: Optional[str]
    iterations: int


def node_router(state: AgentState) -> AgentState:
    task_type = route_task(state["user_message"], state.get("image_path") is not None)
    state["task_type"] = task_type
    state["trace"].append({"step": "router", "detail": f"Classified as: {task_type}"})
    return state


def node_agent(state: AgentState) -> AgentState:
    model_key = "coding" if state["task_type"] == "code" else "reasoning"
    history_text = "\n".join(
        f"- Called {h['action']['tool']}({h['action']['args']}) -> {h['result'][:300]}"
        for h in state["history"]
    ) or "(no actions taken yet)"

    messages = [
        {"role": "system", "content": TOOLS_DESCRIPTION},
        {"role": "user", "content": (
            f"User request: {state['user_message']}\n\n"
            f"Task type: {state['task_type']}\n"
            f"Actions so far:\n{history_text}\n\n"
            f"Decide the single next action as JSON."
        )},
    ]
    raw = model_manager.generate(model_key, messages, max_new_tokens=300)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        action = json.loads(match.group(0)) if match else {"tool": "final_answer", "args": {"answer": raw}}
    except json.JSONDecodeError:
        action = {"tool": "final_answer", "args": {"answer": raw}}

    state["trace"].append({"step": "agent", "detail": f"Plans: {action.get('tool')}({action.get('args')})"})
    state["_pending_action"] = action
    return state


def node_tool(state: AgentState) -> AgentState:
    action = state["_pending_action"]
    tool, args = action.get("tool"), action.get("args", {})

    if tool == "document_search":
        result = tool_document_search(args.get("query", ""))
    elif tool == "file_read":
        result = tool_file_read(args.get("filename", ""))
    elif tool == "file_write":
        result = tool_file_write(args.get("content", ""), args.get("filename", "output"), args.get("fmt", "docx"))
    elif tool == "code_exec":
        result = tool_code_exec(args.get("code", ""))
    elif tool == "vision_ocr":
        image_path = args.get("image_path") or state.get("image_path")
        result = run_vision_ocr(image_path) if image_path else "No image provided."
    elif tool == "final_answer":
        state["final_answer"] = args.get("answer", "")
        result = "final answer produced"
    else:
        result = f"Unknown tool: {tool}"

    state["trace"].append({"step": "tool_call", "tool": tool, "detail": result[:400]})
    state["history"].append({"action": action, "result": result})
    state["iterations"] += 1
    return state


def edge_after_tool(state: AgentState) -> str:
    if state.get("final_answer"):
        return "done"
    if state["iterations"] >= MAX_AGENT_ITERATIONS:
        state["trace"].append({"step": "observe", "detail": "Max iterations reached — returning best-effort summary."})
        summary = "\n".join(h["result"] for h in state["history"][-3:])
        state["final_answer"] = f"(Stopped after {MAX_AGENT_ITERATIONS} steps) Summary of findings:\n{summary}"
        return "done"
    return "continue"


graph = StateGraph(AgentState)
graph.add_node("router", node_router)
graph.add_node("agent", node_agent)
graph.add_node("tool", node_tool)
graph.set_entry_point("router")
graph.add_edge("router", "agent")
graph.add_edge("agent", "tool")
graph.add_conditional_edges("tool", edge_after_tool, {"continue": "agent", "done": END})
compiled_graph = graph.compile()


def run_agent(user_message: str, image_path: Optional[str] = None) -> dict:
    state: AgentState = {
        "user_message": user_message,
        "image_path": image_path,
        "task_type": "",
        "history": [],
        "trace": [],
        "final_answer": None,
        "iterations": 0,
    }
    final_state = compiled_graph.invoke(state)
    return {"answer": final_state.get("final_answer") or "(no answer produced)", "trace": final_state["trace"]}

# =====================================================================
# 8. FASTAPI APP
# =====================================================================
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="VaultMind Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Optional: serve a built frontend from this same process (same-origin,
# no tunnel needed at all). Drop your teammate's `npm run build` output
# into a folder named "frontend_dist" next to this script, and it'll be
# served automatically at "/". If that folder doesn't exist, this is a
# no-op and the API-only behavior is unchanged.
FRONTEND_DIST = "frontend_dist"
if os.path.isdir(FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    print(f"[VaultMind] Serving frontend from ./{FRONTEND_DIST} at the same origin as the API.")


class ChatRequest(BaseModel):
    message: str


@app.get("/health")
def health():
    return {"status": "ok", "lightweight_mode": LIGHTWEIGHT_MODE, "models": MODELS}


@app.get("/network-status")
def network_status():
    return net_monitor.status()


@app.post("/chat")
def chat(message: str = Form(...), image: Optional[UploadFile] = File(None)):
    image_path = None
    if image is not None:
        image_path = os.path.join(SCRATCH_DIR, image.filename)
        with open(image_path, "wb") as f:
            f.write(image.file.read())
    result = run_agent(message, image_path)
    return result


@app.post("/chat-json")
def chat_json(req: ChatRequest):
    """Convenience endpoint for text-only requests (no image), so the
    frontend can send plain JSON instead of multipart form-data."""
    result = run_agent(req.message, None)
    return result


# Optional: serve a built frontend from this same process (same-origin,
# no tunnel needed at all). Drop your teammate's `npm run build` output
# into a folder named "frontend_dist" next to this script, and it'll be
# served automatically at "/". Registered LAST and deliberately — a
# mount at "/" matches every path, so it must come after the specific
# API routes above or it will shadow /health, /chat, etc. entirely.
FRONTEND_DIST = "frontend_dist"
if os.path.isdir(FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
    print(f"[VaultMind] Serving frontend from ./{FRONTEND_DIST} at the same origin as the API.")

# =====================================================================
# 9. STARTUP  (RunPod: plain script, no notebook event-loop juggling
#    needed. No ngrok either — expose port 8000 via RunPod's built-in
#    HTTP port proxy in the pod's network settings, which gives you a
#    public https://<pod-id>-8000.proxy.runpod.net URL with ZERO
#    outbound tunnel connections from inside the pod — cleaner for the
#    "zero external calls" sovereignty proof than ngrok ever was.)
# =====================================================================
def start_server():
    import uvicorn

    build_knowledge_base()
    net_monitor.start()  # start counting AFTER setup/downloads are done

    print(f"\n{'='*60}")
    print("[VaultMind] Starting on 0.0.0.0:8000")
    print("[VaultMind] If using RunPod's HTTP port proxy, your teammate's")
    print("[VaultMind] URL is: https://<your-pod-id>-8000.proxy.runpod.net")
    print(f"{'='*60}\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    start_server()
