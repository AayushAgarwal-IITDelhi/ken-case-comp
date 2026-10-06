"""Bhaiya MCP: one server for the single "Bhaiya" wedding agent (Badhe Bhaiya, Mehendi MVP).

Tools
  gnani_transcribe        voice-note URL -> transcript (Gnani STT)
  gnani_voice_reply       text -> public mp3 URL (Gnani TTS)
  standardize_address     Delhivery Maps mock (real tool name + shape)
  route                   Delhivery Maps mock (real tool name + shape)
  update_wedding_state    write Venue / Photography rows (Wedding State)
  read_wedding_state      read all three tabs
  log_decision            append a Decision Log row (server stamps the time, IST)

Pages
  /        upload a voice note, get a URL to paste into the agent chat
  /state   live view of the Wedding State (Venue, Photography, Decision Log)

No auth by design: this server is for one hackathon demo.
Run: uvicorn server:app --host 0.0.0.0 --port $PORT
"""
import asyncio
import html
import ipaddress
import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

log = logging.getLogger("bhaiya-mcp")
logging.basicConfig(level=logging.INFO)
IST = timezone(timedelta(hours=5, minutes=30))

# ---------------------------------------------------------------- config
GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "")
STT_URL = os.environ.get("GNANI_STT_URL", "https://api.vachana.ai/stt/v3")
TTS_URL = os.environ.get("GNANI_TTS_URL", "https://api.vachana.ai/api/v1/tts/inference")
TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.environ.get("GNANI_TTS_VOICE", "Nalini")
TTS_ENCODING = os.environ.get("GNANI_TTS_ENCODING", "linear_pcm")
TTS_CONTAINER = os.environ.get("GNANI_TTS_CONTAINER", "mp3")
TTS_SAMPLE_RATE = int(os.environ.get("GNANI_TTS_SAMPLE_RATE", "48000"))

MAX_AUDIO_BYTES = 10 * 1024 * 1024
UPLOAD_TTL = 2 * 60 * 60
TTS_TTL = 60 * 60
EXT_MIME = {
    ".ogg": "audio/ogg", ".opus": "audio/ogg", ".oga": "audio/ogg", ".mp3": "audio/mpeg",
    ".wav": "audio/wav", ".m4a": "audio/mp4", ".aac": "audio/aac", ".webm": "audio/webm",
}
MIME_EXT = {"audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/mp4": ".m4a",
            "audio/aac": ".aac", "audio/webm": ".webm", "audio/x-wav": ".wav"}


def _base_url() -> str:
    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if base:
        return base
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    return f"https://{domain}" if domain else ""


def _public_base(request: Request) -> str:
    return _base_url() or f"{request.headers.get('x-forwarded-proto', 'https')}://{request.headers.get('host', '')}"


# ------------------------------------------------- in-memory file store
_files: dict[str, tuple[float, bytes, str]] = {}  # id -> (expires_at, bytes, mime)


def _store(data: bytes, mime: str, ttl: int) -> str:
    now = time.time()
    for k in [k for k, v in _files.items() if v[0] < now]:
        _files.pop(k, None)
    fid = uuid.uuid4().hex
    _files[fid] = (now + ttl, data, mime)
    return fid


def _file_id(name: str) -> str:
    return name.split(".")[0]


# ------------------------------------------------- Wedding State
COLUMNS: dict[str, list[str]] = {
    "Venue": ["raw_description", "standardized_address", "lat", "long", "status"],
    "Photography": ["candidate", "price", "advance", "status", "approved", "payment_status", "feasibility_check"],
    "Decision Log": ["timestamp", "step", "input_received", "input_source", "decision",
                     "rule_followed", "action_taken", "recipient", "action_connector"],
}
STATE: dict[str, list[dict[str, str]]] = {tab: [] for tab in COLUMNS}
_sheet = None
_sync_lock = asyncio.Lock()


def _now() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


def _open_sheet():
    """Open the Google Sheet once. Returns None when Sheets is not configured."""
    global _sheet
    if _sheet is not None:
        return _sheet
    raw, sheet_id = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON"), os.environ.get("GOOGLE_SHEET_ID")
    if not raw or not sheet_id:
        return None
    import gspread

    sh = gspread.service_account_from_dict(json.loads(raw)).open_by_key(sheet_id)
    for tab, cols in COLUMNS.items():
        try:
            sh.worksheet(tab)
        except gspread.WorksheetNotFound:
            sh.add_worksheet(tab, rows=200, cols=len(cols))
    _sheet = sh
    return sh


def _push_tab_blocking(tab: str):
    sh = _open_sheet()
    if sh is None:
        return None
    values = [COLUMNS[tab]] + [[r.get(c, "") for c in COLUMNS[tab]] for r in STATE[tab]]
    ws = sh.worksheet(tab)
    ws.clear()
    ws.update(values=values, range_name="A1")
    return True


async def _sync(tab: str) -> str:
    """Mirror one tab to Google Sheets. Never raises: the in-memory state is the source of truth."""
    async with _sync_lock:
        try:
            res = await asyncio.to_thread(_push_tab_blocking, tab)
        except Exception as exc:  # noqa: BLE001
            log.warning("Sheets sync failed for %s: %s", tab, exc)
            return "failed"
    return "ok" if res else "not_configured"


# ------------------------------------------------- server
# host="0.0.0.0" keeps the SDK from enabling localhost-only host-header protection.
mcp = FastMCP("bhaiya-tools", host="0.0.0.0", stateless_http=True, json_response=True)


def _require_key() -> None:
    if not GNANI_API_KEY:
        raise RuntimeError("GNANI_API_KEY is not configured on the server")


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return False
    return True


async def _fetch_audio(url: str) -> tuple[bytes, str]:
    """Get audio bytes from our own upload store or a public http(s) URL (redirects re-checked)."""
    base = _base_url()
    if base and url.startswith(base + "/files/"):
        item = _files.get(_file_id(url.rsplit("/", 1)[1]))
        if not item or item[0] < time.time():
            raise ValueError("That uploaded voice note has expired. Upload it again.")
        return item[1], item[2]
    async with httpx.AsyncClient(timeout=30) as client:
        current = url
        for _ in range(4):
            p = urlparse(current)
            if p.scheme not in ("http", "https") or not p.hostname:
                raise ValueError("audio_url must be an http(s) URL")
            if not await asyncio.to_thread(_host_is_public, p.hostname):
                raise ValueError("audio_url host is not allowed")
            r = await client.get(current, follow_redirects=False)
            if r.is_redirect:
                current = urljoin(current, r.headers["location"])
                continue
            r.raise_for_status()
            if len(r.content) > MAX_AUDIO_BYTES:
                raise ValueError("audio too large (10 MB max)")
            return r.content, r.headers.get("content-type", "audio/ogg").split(";")[0]
    raise ValueError("too many redirects")


# ---------------------------------------------------------- Gnani tools
@mcp.tool()
async def gnani_transcribe(audio_url: str, language_code: str = "hi-IN") -> dict:
    """Transcribe a short voice note (60 seconds or less) to text with Gnani speech-to-text.

    audio_url: https URL of the voice note (the URL the family gave, or one from this server's upload page).
    language_code: BCP-47 code such as hi-IN (Hindi) or en-IN (English).
    Returns {"transcript": str}. Report unclear audio honestly; never fill gaps with guesses.
    """
    _require_key()
    audio, mime = await _fetch_audio(audio_url)
    ext = MIME_EXT.get(mime, ".ogg")
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            STT_URL,
            headers={"X-API-Key-ID": GNANI_API_KEY},
            files={"audio_file": (f"voice_note{ext}", audio, mime)},
            data={"language_code": language_code, "format": "transcribe"},
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Gnani STT failed: HTTP {resp.status_code} {resp.text[:300]}")
    body = resp.json()
    if not body.get("success", True) or not body.get("transcript"):
        raise RuntimeError(f"Gnani STT returned no transcript: {str(body)[:300]}")
    return {"transcript": body["transcript"], "language_code": language_code}


@mcp.tool()
async def gnani_voice_reply(text: str, language: str = "hi-IN", voice: str = "") -> dict:
    """Convert text to speech with Gnani and return a public mp3 link (audio_url).

    Speaks exactly the text given. Put the returned audio_url in your chat reply so the family can play it.
    Only voice text that is already approved to be said to the family.
    """
    _require_key()
    base = _base_url()
    if not base:
        raise RuntimeError("PUBLIC_BASE_URL is not configured on the server")
    if not text.strip() or len(text) > 2000:
        raise ValueError("text must be 1-2000 characters")
    payload = {
        "text": text, "voice": voice or TTS_VOICE, "model": TTS_MODEL, "language": language, "speed": 1,
        "audio_config": {"encoding": TTS_ENCODING, "container": TTS_CONTAINER, "num_channels": 1,
                         "sample_rate": TTS_SAMPLE_RATE, "sample_width": 2},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(TTS_URL, headers={"X-API-Key-ID": GNANI_API_KEY}, json=payload)
    if resp.status_code != 200 or not resp.content:
        raise RuntimeError(f"Gnani TTS failed: HTTP {resp.status_code} {resp.text[:300]}")
    mime = {"ogg": "audio/ogg", "wav": "audio/wav", "mp3": "audio/mpeg"}.get(
        TTS_CONTAINER, resp.headers.get("content-type", "application/octet-stream").split(";")[0])
    fid = _store(resp.content, mime, TTS_TTL)
    return {"audio_url": f"{base}/files/{fid}{MIME_EXT.get(mime, '')}", "mime_type": mime}


# ------------------------------------------- Delhivery Maps mock tools
@mcp.tool()
async def standardize_address(address: str) -> dict:
    """Parse a messy, free-form Indian address into a standardized address with pincode, locality,
    city, state, latitude and longitude, plus a confidence score (Delhivery Maps).

    Pass the family's venue description exactly as given, unedited.
    Demo/test behavior (mock):
      - empty address -> error MALFORMED_REQUEST
      - contains "xxfail" -> error ADDRESS_NOT_RESOLVABLE
      - contains "xxslow" -> deliberate 6 s delay (timeout test)
      - contains "chhattarpur" -> resolves to the Sharma family's Mehendi venue (confidence 0.86)
      - anything else -> low-confidence (0.4) generic fallback
    """
    if not address or not address.strip():
        return {"status": "ERROR", "error_code": "MALFORMED_REQUEST",
                "message": "The 'address' field is required and must be non-empty text."}
    lowered = address.lower()
    if "xxslow" in lowered:
        await asyncio.sleep(6)
    if "xxfail" in lowered:
        return {"status": "ERROR", "error_code": "ADDRESS_NOT_RESOLVABLE",
                "message": "The address could not be standardized: too incomplete or ambiguous to resolve confidently."}
    if "chhattarpur" in lowered:
        return {
            "status": "OK",
            "standardized_address": {
                "formatted_address": "Green Leaf Farms, Chhattarpur Enclave, Near Chhattarpur Mandir Metro Station, New Delhi, Delhi 110074",
                "locality": "Chhattarpur Enclave", "city": "New Delhi", "state": "Delhi", "pincode": "110074",
                "latitude": 28.5011, "longitude": 77.1727,
            },
            "confidence": 0.86, "source": "GeoNaksha LLM (mocked)",
        }
    return {
        "status": "OK",
        "standardized_address": {"formatted_address": address.strip(), "locality": None, "city": None,
                                 "state": None, "pincode": None, "latitude": None, "longitude": None},
        "confidence": 0.4, "source": "GeoNaksha LLM (mocked)",
    }


@mcp.tool()
async def route(coordinates: list[list[float]], travel_mode: str = "auto") -> dict:
    """Calculate a driving route and travel time between two or more [latitude, longitude] waypoints
    (Delhivery Maps). Returns distance_km and duration_minutes in current traffic.

    Demo/test behavior (mock):
      - fewer than 2 coordinate pairs -> error MALFORMED_REQUEST
      - a [0, 0] pair -> error UNROUTABLE_LOCATION
      - travel_mode "xxslow" -> deliberate 6 s delay (timeout test)
      - otherwise -> hardcoded demo result (14.2 km, 52 minutes: LensCraft Studios, Lajpat Nagar, to the Chhattarpur venue)
    """
    if not coordinates or len(coordinates) < 2:
        return {"status": "ERROR", "error_code": "MALFORMED_REQUEST",
                "message": "At least two [lat, lng] coordinate pairs are required."}
    if travel_mode == "xxslow":
        await asyncio.sleep(6)
    if any(lat == 0 and lng == 0 for lat, lng in coordinates):
        return {"status": "ERROR", "error_code": "UNROUTABLE_LOCATION",
                "message": "One or more coordinates could not be matched to a road network."}
    return {"status": "OK", "distance_km": 14.2, "duration_minutes": 52, "traffic_aware": True,
            "vehicle_profile": travel_mode}


# --------------------------------------------------- Wedding State tools
@mcp.tool()
async def update_wedding_state(tab: Literal["Venue", "Photography"], fields: dict[str, Any]) -> dict:
    """Write to the Wedding State (a Google Sheet mirror). Call this after each fact is established.

    tab "Venue" (one row): raw_description, standardized_address, lat, long, status.
    tab "Photography" (one row per photographer; "candidate" is required and identifies the row):
      candidate, price, advance, status, approved, payment_status, feasibility_check.
    status is "raw" until a tool verifies it, then "verified". feasibility_check is ok, risk or unchecked.
    Only the fields you pass are changed. Unknown field names are rejected.
    """
    allowed = COLUMNS[tab]
    unknown = [k for k in fields if k not in allowed]
    if unknown:
        return {"ok": False, "error": f"Unknown fields {unknown} for tab {tab}. Allowed: {allowed}"}
    clean = {k: "" if v is None else str(v) for k, v in fields.items()}
    if tab == "Venue":
        if not STATE["Venue"]:
            STATE["Venue"].append({})
        row = STATE["Venue"][0]
    else:
        cand = clean.get("candidate", "").strip()
        if not cand:
            return {"ok": False, "error": "Photography rows need a 'candidate' field."}
        row = next((r for r in STATE["Photography"] if r.get("candidate", "").lower() == cand.lower()), None)
        if row is None:
            row = {}
            STATE["Photography"].append(row)
    row.update(clean)
    return {"ok": True, "tab": tab, "row": row, "sheet_sync": await _sync(tab)}


@mcp.tool()
async def read_wedding_state() -> dict:
    """Read the whole Wedding State: Venue, Photography and Decision Log rows."""
    return {tab: STATE[tab] for tab in COLUMNS}


@mcp.tool()
async def log_decision(step: str, input_received: str, input_source: str, decision: str, rule_followed: str,
                       action_taken: str, recipient: str = "None", action_connector: str = "None") -> dict:
    """Append one row to the Decision Log. Call this once for EVERY decision you make, right after making it.

    step: intake / venue / shortlist / feasibility / approval / vendor contact / payment.
    input_received: what triggered the decision (quote it). input_source: connector and real source behind it.
    decision: what you chose over the alternative. rule_followed: the rule in your instructions.
    action_taken: the actual message or action, word for word. recipient: who it went to, or "None".
    action_connector: the connector used to act, or "None". The server stamps the time (IST).
    """
    row = {"timestamp": _now(), "step": step, "input_received": input_received, "input_source": input_source,
           "decision": decision, "rule_followed": rule_followed, "action_taken": action_taken,
           "recipient": recipient, "action_connector": action_connector}
    STATE["Decision Log"].append(row)
    return {"ok": True, "timestamp": row["timestamp"], "row_number": len(STATE["Decision Log"]),
            "sheet_sync": await _sync("Decision Log")}


# ------------------------------------------------------------- web pages
UPLOAD_PAGE = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bhaiya: upload a voice note</title>
<style>body{font-family:system-ui,sans-serif;max-width:560px;margin:40px auto;padding:0 16px}code{word-break:break-all}</style>
<h1>Upload a voice note</h1>
<p>Pick the family's voice note (.ogg, .opus, .mp3, .wav, 10 MB max). You get a URL to paste into the agent chat.
Live Wedding State: <a href="/state">/state</a></p>
<input id="f" type="file" accept="audio/*,.opus,.ogg"><p id="out"></p>
<script>
document.getElementById('f').onchange = async (e) => {
  const file = e.target.files[0]; if (!file) return;
  const out = document.getElementById('out'); out.textContent = 'Uploading...';
  try {
    const r = await fetch('/upload', {method: 'POST',
      headers: {'X-Filename': file.name, 'Content-Type': file.type || 'application/octet-stream'}, body: file});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
    out.innerHTML = 'URL (valid 2 hours):<br><code>' + j.url + '</code><br><br><audio controls src="' + j.url + '"></audio>';
  } catch (err) { out.textContent = 'Failed: ' + err.message; }
};
</script>"""


def _table(tab: str) -> str:
    cols = COLUMNS[tab]
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    rows = "".join("<tr>" + "".join(f"<td>{html.escape(r.get(c, ''))}</td>" for c in cols) + "</tr>"
                   for r in STATE[tab])
    if not rows:
        rows = f'<tr><td colspan="{len(cols)}" style="color:#888">no rows yet</td></tr>'
    return f"<h2>{html.escape(tab)}</h2><div style='overflow-x:auto'><table><tr>{head}</tr>{rows}</table></div>"


@mcp.custom_route("/", methods=["GET"])
async def home(_: Request) -> Response:
    return HTMLResponse(UPLOAD_PAGE)


@mcp.custom_route("/health", methods=["GET"])
async def health(_: Request) -> Response:
    return JSONResponse({"ok": True, "service": "bhaiya-mcp", "sheets": bool(os.environ.get("GOOGLE_SHEET_ID"))})


@mcp.custom_route("/state", methods=["GET"])
async def state_page(_: Request) -> Response:
    page = ("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Wedding State</title><style>body{font-family:system-ui,sans-serif;margin:24px}"
            "table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;"
            "vertical-align:top}th{background:#f3f3f3}</style><h1>Wedding State</h1>"
            "<p><button onclick=\"if(confirm('Clear all rows?'))fetch('/reset',{method:'POST'}).then(()=>location.reload())\">"
            "Reset for a new run</button> (auto-refreshes every 4 s)</p>"
            + "".join(_table(t) for t in COLUMNS) + "<script>setTimeout(()=>location.reload(),4000)</script>")
    return HTMLResponse(page)


@mcp.custom_route("/reset", methods=["POST"])
async def reset(_: Request) -> Response:
    for tab in COLUMNS:
        STATE[tab].clear()
        await _sync(tab)
    return JSONResponse({"ok": True})


@mcp.custom_route("/upload", methods=["POST"])
async def upload(request: Request) -> Response:
    data = await request.body()
    if not data:
        return JSONResponse({"error": "empty file"}, status_code=400)
    if len(data) > MAX_AUDIO_BYTES:
        return JSONResponse({"error": "file too large (10 MB max)"}, status_code=413)
    ext = os.path.splitext(request.headers.get("x-filename", ""))[1].lower()
    mime = EXT_MIME.get(ext) or request.headers.get("content-type", "").split(";")[0] or "application/octet-stream"
    if not ext:
        ext = MIME_EXT.get(mime, "")
    fid = _store(data, mime, UPLOAD_TTL)
    return JSONResponse({"url": f"{_public_base(request)}/files/{fid}{ext}", "expires_in_seconds": UPLOAD_TTL})


@mcp.custom_route("/files/{name}", methods=["GET"])
async def get_file(request: Request) -> Response:
    item = _files.get(_file_id(request.path_params["name"]))
    if not item or item[0] < time.time():
        return Response(status_code=404)
    return Response(content=item[1], media_type=item[2])


app = mcp.streamable_http_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
