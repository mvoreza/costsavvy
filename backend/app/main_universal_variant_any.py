# app/main.py
import os
import json
import re
import uuid
import time
import logging
from typing import Optional, Any, Dict, List, Tuple

import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI, OpenAIError

# Python fallback refiners (file: app/service_refiners.py)
from app.service_refiners import refiners_registry

# ----------------------------
# Config
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("costsavvy")

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

APP_API_KEY = os.getenv("APP_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
MIN_DB_RESULTS_BEFORE_WEB = int(os.getenv("MIN_DB_RESULTS_BEFORE_WEB", "3"))

# Always try to show at least this many facilities in the answer
MIN_FACILITIES_TO_DISPLAY = int(os.getenv("MIN_FACILITIES_TO_DISPLAY", "5"))

# Refiners cache TTL (DB-first, Python fallback)
REFINERS_CACHE_TTL_SECONDS = int(os.getenv("REFINERS_CACHE_TTL_SECONDS", "300"))

# ---- Intent override controls (env-driven) ----
INTENT_OVERRIDE_FORCE_PRICE_ENABLED = os.getenv("INTENT_OVERRIDE_FORCE_PRICE_ENABLED", "true").lower() in (
    "1", "true", "yes", "y", "on"
)
INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS = [
    s.strip().lower()
    for s in os.getenv(
        "INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS",
        "cost,price,how much,pricing,estimate,rate,charge,fee",
    ).split(",")
    if s.strip()
]

# For new pricing questions, do NOT reuse old ZIP/payment fields from session
RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION = os.getenv("RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION", "true").lower() in (
    "1", "true", "yes", "y", "on"
)

# Progressive radius attempts for priced search (miles)
PRICE_RADIUS_ATTEMPTS = [
    int(x) for x in os.getenv("PRICE_RADIUS_ATTEMPTS", "10,25,50,100,200").split(",") if x.strip().isdigit()
] or [10, 25, 50, 100, 200]

# ----------------------------
# App
# ----------------------------
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)
pool: asyncpg.Pool | None = None

app.mount("/static", StaticFiles(directory="static"), name="static")


# ----------------------------
# Models
# ----------------------------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


# ----------------------------
# Rate limiting
# ----------------------------
_ip_hits: Dict[str, List[float]] = {}


def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    window_start = now - 60
    hits = _ip_hits.get(ip, [])
    hits = [t for t in hits if t >= window_start]
    if len(hits) >= RATE_LIMIT_PER_MINUTE:
        _ip_hits[ip] = hits
        return False
    hits.append(now)
    _ip_hits[ip] = hits
    return True


def require_auth(request: Request):
    if not APP_API_KEY:
        return
    if request.headers.get("X-API-Key", "") != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


# ----------------------------
# Startup / Shutdown
# ----------------------------
@app.on_event("startup")
async def startup():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)


@app.on_event("shutdown")
async def shutdown():
    if pool:
        await pool.close()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def home():
    return FileResponse("static/index.html")


# ----------------------------
# SSE helpers
# ----------------------------
def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def stream_llm_to_sse(system: str, user_content: str, out_text_parts: List[str]):
    """
    Streams OpenAI chat.completions deltas as SSE 'delta' events.
    """
    try:
        stream = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            stream=True,
            timeout=30,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                out_text_parts.append(delta)
                yield sse({"type": "delta", "text": delta})
    except OpenAIError as e:
        logger.error(f"OpenAI error: {e}")
        yield sse({"type": "error", "message": f"OpenAI error: {type(e).__name__}"})
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield sse({"type": "error", "message": f"Streaming error: {type(e).__name__}"})


# ----------------------------
# DB helpers
# ----------------------------
def _coerce_jsonb_to_dict(val) -> dict:
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


async def get_or_create_session(conn: asyncpg.Connection, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    if not session_id:
        session_id = str(uuid.uuid4())
    row = await conn.fetchrow("SELECT session_state FROM public.chat_session WHERE id = $1", session_id)
    if row:
        await conn.execute("UPDATE public.chat_session SET last_seen = now() WHERE id = $1", session_id)
        return session_id, _coerce_jsonb_to_dict(row["session_state"])
    await conn.execute(
        "INSERT INTO public.chat_session (id, session_state) VALUES ($1, $2::jsonb)",
        session_id,
        json.dumps({}),
    )
    return session_id, {}


async def save_message(conn: asyncpg.Connection, session_id: str, role: str, content: str, metadata: Optional[dict] = None):
    await conn.execute(
        "INSERT INTO public.chat_message (session_id, role, content, metadata) VALUES ($1,$2,$3,$4::jsonb)",
        session_id,
        role,
        content,
        json.dumps(metadata or {}),
    )


async def update_session_state(conn: asyncpg.Connection, session_id: str, state: Dict[str, Any]):
    await conn.execute(
        "UPDATE public.chat_session SET session_state = $2::jsonb, last_seen = now() WHERE id = $1",
        session_id,
        json.dumps(state),
    )


async def log_query(
    conn: asyncpg.Connection,
    session_id: str,
    question: str,
    intent: dict,
    used_radius: Optional[float],
    result_count: int,
    used_web: bool,
    answer_text: str,
):
    await conn.execute(
        """
        INSERT INTO public.query_log
          (session_id, question, intent_json, used_radius_miles, result_count, used_web_search, answer_text)
        VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7)
        """,
        session_id,
        question,
        json.dumps(intent),
        used_radius,
        result_count,
        used_web,
        answer_text,
    )


# ----------------------------
# Intent extraction
# ----------------------------
INTENT_RULES = """
Return ONLY JSON with:
mode: "general" | "price" | "hybrid" | "clarify"
zipcode: 5-digit ZIP or null
radius_miles: number or null
payer_like: string like "Aetna" or null
plan_like: string like "PPO" or null
payment_mode: "cash" | "insurance" | null
service_query: short phrase like "chest x-ray" or null
code_type: string or null (usually "CPT")
code: string or null (like "71046")
clarifying_question: string or null
cash_only: boolean

Notes:
- If user says "cash price", "self-pay", "out of pocket" => payment_mode="cash" and cash_only=true.
- If user mentions insurance or a carrier name => payment_mode="insurance".
"""


def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in ["zipcode", "radius_miles", "payer_like", "plan_like", "payment_mode", "service_query", "code_type", "code", "cash_only"]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


def _is_zip_only_message(msg: str) -> bool:
    s = (msg or "").strip()
    return len(s) == 5 and s.isdigit()


def _normalize_payment_mode(merged: Dict[str, Any]) -> None:
    if merged.get("cash_only") is True:
        merged["payment_mode"] = "cash"
        merged["payer_like"] = None
        merged["plan_like"] = None
        return

    pm = merged.get("payment_mode")
    if pm == "cash":
        merged["cash_only"] = True
        merged["payer_like"] = None
        merged["plan_like"] = None
    elif pm == "insurance":
        merged["cash_only"] = False


async def extract_intent(message: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic session-aware intent extraction.

    Key goals:
      - Keep the user in the pricing flow once they started it.
      - Ask for ZIP first if missing (geo-first).
      - Accept short replies like "Cash", "Self pay", "Insurance", "Aetna" as continuations.
      - Allow the user to change ZIP/payment/payer mid-session and rerun pricing accordingly.
    """
    msg = (message or "").strip()
    msg_l = msg.lower()

    # Pull existing context
    st = state or {}
    awaiting = st.get("_awaiting")
    have_service = bool(st.get("service_query") or st.get("code"))
    have_zip = bool(st.get("zipcode"))

    # 0) ZIP detection anywhere in the message
    zip_match = re.search(r"\b(\d{5})\b", msg)
    if zip_match:
        z = zip_match.group(1)
        # If we already have a service in session, treat this as continuation of pricing flow.
        if have_service or awaiting in {"zip", "payment", "payer"}:
            return {
                "mode": "price",
                "zipcode": z,
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": st.get("payment_mode"),
                "payer_like": st.get("payer_like"),
                "plan_like": st.get("plan_like"),
                "clarifying_question": None,
            }

    # Helpers
    cash_terms = {
        "cash", "self pay", "self-pay", "selfpay", "out of pocket", "oop",
        "paying cash", "pay cash", "cash pay", "self pay patient", "selfpay patient",
    }
    insurance_terms = {"insurance", "insured", "use insurance", "with insurance"}

    carrier_map = {
        "aetna": "Aetna",
        "cigna": "Cigna",
        "anthem": "Anthem",
        "blue cross": "Blue Cross Blue Shield",
        "bcbs": "Blue Cross Blue Shield",
        "united": "UnitedHealthcare",
        "uhc": "UnitedHealthcare",
        "humana": "Humana",
        "kaiser": "Kaiser Permanente",
        "molina": "Molina",
        "centene": "Centene",
        "wellcare": "Wellcare",
        "medicaid": "Medicaid",
        "medicare": "Medicare",
    }

    def extract_carrier(m: str) -> Optional[str]:
        ml = (m or "").lower()
        # direct map matches
        for k, v in carrier_map.items():
            if k in ml:
                return v

        # Filter out common filler words so we don't accidentally treat
        # "I have insurance" or "use insurance" as the carrier name "I Have Insurance".
        clean = m
        for stop in ["i have", "i use", "use", "with", "insurance", "my", "have", "paying"]:
            # Word-boundary aware case-insensitive replacement
            clean = re.sub(r'\b' + re.escape(stop) + r'\b', '', clean, flags=re.IGNORECASE)

        clean = clean.strip()
        tokens = re.findall(r"[a-zA-Z]+", clean)

        # if the user replies with just a single word, treat it as a carrier candidate
        if len(tokens) == 1 and len(tokens[0]) >= 3:
            return tokens[0].title()
        
        # two-word carrier (e.g., "Harvard Pilgrim")
        if 2 <= len(tokens) <= 3:
            return " ".join([t.title() for t in tokens])
            
        return None

    # 1) If we are explicitly awaiting ZIP, only accept ZIP (or a cancel/new question handled by LLM below).
    if awaiting == "zip":
        # If they didn't provide a ZIP, keep asking.
        return {"mode": "clarify", "clarifying_question": "What’s your 5-digit ZIP code?"}

    # 2) If we are awaiting payment, accept cash/insurance quickly.
    if awaiting == "payment":
        if any(t in msg_l for t in cash_terms):
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "cash",
                "payer_like": None,
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": True,
            }

        if any(t in msg_l for t in insurance_terms) or extract_carrier(msg):
            payer_like = extract_carrier(msg)
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "insurance",
                "payer_like": payer_like,
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": False,
            }

        return {"mode": "clarify", "clarifying_question": "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"}

    # 3) Session-aware continuation even when not awaiting:
    # If the session already has service + zip, treat "cash"/"insurance"/carrier-only as updates and rerun pricing.
    if have_service and have_zip:
        if any(t in msg_l for t in cash_terms):
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "cash",
                "payer_like": None,
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": True,
            }

        carrier = extract_carrier(msg)
        if any(t in msg_l for t in insurance_terms) or carrier:
            return {
                "mode": "price",
                "zipcode": st.get("zipcode"),
                "service_query": st.get("service_query"),
                "code_type": st.get("code_type"),
                "code": st.get("code"),
                "payment_mode": "insurance",
                "payer_like": carrier or st.get("payer_like"),
                "plan_like": None,
                "clarifying_question": None,
                "cash_only": False,
            }

    # 4) If the user is asking a new price question but we lack ZIP, we will ask for ZIP (geo-first).
    inferred_service = infer_service_query_from_message(msg)
    if inferred_service and not have_zip:
        # Force clarifying ZIP instead of letting the LLM ask for payment first.
        return {"mode": "price", "service_query": inferred_service, "clarifying_question": None, "_new_price_question": True}

    # 5) Otherwise fall back to LLM-based intent extraction (general Q&A, non-price).
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "You extract intent for healthcare Q&A and price lookup. Be conservative."},
                {"role": "system", "content": INTENT_RULES},
                {"role": "user", "content": json.dumps({"message": msg, "session_state": st})},
            ],
            timeout=10,
        )
        content = resp.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as e:
        logger.warning(f"Intent extraction failed: {e}")
        return {"mode": "clarify", "clarifying_question": "What 5-digit ZIP code should I search near?"}

def infer_service_query_from_message(message: str) -> Optional[str]:
    msg = (message or "").lower()
    if "colonoscopy" in msg:
        return "colonoscopy"
    if "mammogram" in msg or "mammo" in msg:
        return "mammogram"
    if "ultrasound" in msg:
        return "ultrasound"
    if "cat scan" in msg or "ct scan" in msg or (" ct " in f" {msg} "):
        return "ct scan"
    if "mri" in msg:
        return "mri"
    if "x-ray" in msg or "xray" in msg:
        return "x-ray"
    if "blood test" in msg or "lab test" in msg or "labs" in msg:
        return "lab test"
    if "office visit" in msg or "doctor visit" in msg:
        return "office visit"
    return None


def should_force_price_mode(message: str, merged: Dict[str, Any]) -> bool:
    if not INTENT_OVERRIDE_FORCE_PRICE_ENABLED:
        return False

    # If we already have a pricing context (service + ZIP) and are still collecting
    # payment details, keep the conversation in price mode even if the user's reply
    # doesn't include explicit cost keywords (e.g., "I have Aetna insurance").
    if (merged or {}).get("service_query") and (merged or {}).get("zipcode") and not (merged or {}).get("payment_mode"):
        return True
    if (merged or {}).get("_awaiting") in {"zip", "payment"}:
        return True

    msg = (message or "").lower()
    return any(kw in msg for kw in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS)


def apply_intent_override_if_needed(intent: Dict[str, Any], message: str, merged: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    if should_force_price_mode(message, merged):
        prev = intent.get("mode")
        if prev != "price":
            logger.info(
                "Intent override applied: forcing mode=price",
                extra={"session_id": session_id, "prev_mode": prev, "keywords": INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS},
            )
        intent["mode"] = "price"
        intent["intent_overridden"] = True
        intent["override_reason"] = "cost_keyword"

        if not (merged.get("service_query") or "").strip():
            inferred = infer_service_query_from_message(message)
            if inferred:
                merged["service_query"] = inferred
                logger.info("Service query inferred from message", extra={"session_id": session_id, "service_query": inferred})
    else:
        intent["intent_overridden"] = False

    return intent


def message_contains_zip(message: str) -> bool:
    msg = (message or "").strip()
    tokens = [t.strip(",.()[]{}") for t in msg.split()]
    return any(len(t) == 5 and t.isdigit() for t in tokens)


def message_contains_payment_info(message: str) -> bool:
    msg = (message or "").lower()
    cash_terms = ["cash", "self pay", "self-pay", "out of pocket", "out-of-pocket", "uninsured", "no insurance"]
    ins_terms = ["insurance", "insured", "copay", "coinsurance", "deductible"]
    return any(t in msg for t in cash_terms) or any(t in msg for t in ins_terms)


def reset_gating_fields_for_new_price_question(message: str, merged: Dict[str, Any]) -> None:
    if not message_contains_zip(message):
        merged.pop("zipcode", None)

    if not message_contains_payment_info(message):
        merged.pop("payment_mode", None)
        merged.pop("payer_like", None)
        merged.pop("plan_like", None)
        merged.pop("cash_only", None)


# ----------------------------
# DB: resolve code + price lookup
# ----------------------------

# ----------------------------
# DB: service_variants matching (universal variant-first)
# ----------------------------
async def _detect_service_variants_columns(conn: asyncpg.Connection) -> Dict[str, bool]:
    cols = {"variant_name": False, "cpt_explanation": False, "patient_summary": False, "patient_summar": False, "cpt_code": False}
    try:
        rows = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='service_variants'
            """
        )
        have = {r["column_name"] for r in rows}
        for k in list(cols.keys()):
            cols[k] = k in have
    except Exception:
        pass
    return cols


async def get_service_variants_by_text(conn: asyncpg.Connection, user_text: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Search public.service_variants for variants matching the user's description (variant_name/cpt_explanation)."""
    text = (user_text or "").strip()
    if not text:
        return []

    cols = await _detect_service_variants_columns(conn)
    name_col = "variant_name" if cols.get("variant_name") else None
    expl_col = "cpt_explanation" if cols.get("cpt_explanation") else None
    # patient_summary may have a historical misspelling
    if cols.get("patient_summary"):
        summary_expr = "patient_summary"
    elif cols.get("patient_summar"):
        summary_expr = "patient_summar AS patient_summary"
    else:
        summary_expr = "NULL::text AS patient_summary"

    if not name_col and not expl_col:
        return []

    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(t) >= 3][:8]
    full_pat = f"%{text.lower()}%"
    pats = [f"%{t}%" for t in tokens]

    params: List[Any] = [full_pat] + pats
    def ph(i: int) -> str:
        return f"${i}"

    ors = []
    # full text match
    if name_col:
        ors.append(f"lower({name_col}) LIKE {ph(1)}")
    if expl_col:
        ors.append(f"lower({expl_col}) LIKE {ph(1)}")
    # token matches
    for i in range(2, len(params) + 1):
        if name_col:
            ors.append(f"lower({name_col}) LIKE {ph(i)}")
        if expl_col:
            ors.append(f"lower({expl_col}) LIKE {ph(i)}")

    where_sql = " OR ".join(ors) if ors else "FALSE"

    score_sql = "CASE WHEN (" + (" OR ".join([f"lower({c}) LIKE $1" for c in [name_col, expl_col] if c])) + ") THEN 100 ELSE 0 END"
    for i in range(2, len(params) + 1):
        score_sql += " + CASE WHEN (" + (" OR ".join([f"lower({c}) LIKE {ph(i)}" for c in [name_col, expl_col] if c])) + ") THEN 5 ELSE 0 END"

    rows = await conn.fetch(
        f"""
        SELECT
          id,
          COALESCE(parent_service, '') AS parent_service,
          {("cpt_code" if cols.get("cpt_code") else "NULL::text AS cpt_code")},
          {("variant_name" if name_col else "NULL::text AS variant_name")},
          {summary_expr},
          COALESCE(is_preventive, FALSE) AS is_preventive,
          {("cpt_explanation" if expl_col else "NULL::text AS cpt_explanation")},
          ({score_sql}) AS score
        FROM public.service_variants
        WHERE {where_sql}
        ORDER BY score DESC, variant_name ASC, id ASC
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [dict(r) for r in rows]


def _simplify_variant_for_patient(v: Dict[str, Any]) -> str:
    name = (v.get("variant_name") or "").strip()
    expl = (v.get("cpt_explanation") or "").strip()
    summary = (v.get("patient_summary") or "").strip()

    base = summary or expl
    base = re.sub(r"\s+", " ", base).strip()
    if len(base) > 160:
        base = base[:157].rstrip() + "..."

    low = (name + " " + base).lower()
    hints = []
    if "without contrast" in low or "w/o contrast" in low:
        hints.append("no IV dye")
    if "with contrast" in low or "w/ contrast" in low:
        hints.append("uses IV dye")
    if "view" in low:
        hints.append("view count matters")
    if "screen" in low and "colon" in low:
        hints.append("routine screening")
    if "diagn" in low:
        hints.append("for symptoms/abnormal results")
    hint = (" (" + ", ".join(hints) + ")") if hints else ""
    return (base + hint) if base else ("Details affect preparation and price." + hint)


def build_variant_options_prompt(service_label: str, variants: List[Dict[str, Any]]) -> str:
    label = (service_label or "this service").strip()
    lines = [f"Before I look up prices, which specific **{label.upper()}** do you mean?", ""]
    for i, v in enumerate(variants, start=1):
        vname = (v.get("variant_name") or "Option").strip()
        desc = _simplify_variant_for_patient(v)
        lines.append(f"{i}) {vname} – {desc}")
    lines.append("")
    lines.append("Reply with the number that best matches what you need.")
    return "\n".join(lines)


def build_variant_confirm_prompt(v: Dict[str, Any]) -> str:
    vname = (v.get("variant_name") or "").strip() or "that option"
    return f"Just to confirm, you mean: **{vname}**. Is that right? (yes/no)"


async def resolve_service_code(conn: asyncpg.Connection, merged: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    if merged.get("code_type") and merged.get("code"):
        return merged["code_type"], merged["code"]

    q = (merged.get("service_query") or "").strip()
    if not q:
        return None

    # 1) Canonical lookup from `services`
    rows = await conn.fetch(
        """
        SELECT code_type, code
        FROM public.services
        WHERE (cpt_explanation ILIKE '%' || $1 || '%'
            OR service_description ILIKE '%' || $1 || '%'
            OR patient_summary ILIKE '%' || $1 || '%')
        ORDER BY code_type, code
        LIMIT 5
        """,
        q,
    )
    if rows:
        return rows[0]["code_type"], rows[0]["code"]

    # 2) Fallback: staged hospital file often has the right CPT even when `services` lacks the keyword
    srows = await conn.fetch(
        """
        SELECT code_type, code, COUNT(*) AS n
        FROM public.stg_hospital_rates
        WHERE code_type IS NOT NULL AND code IS NOT NULL
          AND (
                service_description ILIKE '%' || $1 || '%'
             OR code ILIKE '%' || $1 || '%'
          )
        GROUP BY code_type, code
        ORDER BY n DESC, code_type, code
        LIMIT 5
        """,
        q,
    )
    if srows:
        return srows[0]["code_type"], srows[0]["code"]

    return None

async def price_lookup_v3(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    radius_array: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    radius_array = radius_array or [10, 25, 50]
    rows = await conn.fetch(
        """
        SELECT *
        FROM public.get_prices_by_zip_radius_v3(
          $1, $2, $3, $4, $5,
          $6::int[], 10, 25
        );
        """,
        zipcode,
        code_type,
        code,
        payer_like,
        plan_like,
        radius_array,
    )
    return [dict(r) for r in rows]


async def price_lookup_staging(
    conn: asyncpg.Connection,
    zipcode: str,
    service_query: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """
    DB-first fallback when we cannot resolve a code or the pricing function returns no rows.
    It searches `stg_hospital_rates` by service_description and joins to `hospitals` + `zip_locations`
    to compute distance (when possible). Returns a row shape compatible with `build_facility_block`.
    """
    q = (service_query or "").strip()
    if not q:
        return []

    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1",
        zipcode,
    )
    zlat = float(z["latitude"]) if z and z["latitude"] is not None else None
    zlon = float(z["longitude"]) if z and z["longitude"] is not None else None

    where_bits = ["r.service_description ILIKE '%' || $2 || '%'"]
    args = [zipcode, q, payer_like, plan_like, limit]
    # payer/plan filters are optional
    payer_clause = "($3::text IS NULL OR r.payer_name ILIKE '%' || $3 || '%')"
    plan_clause = "($4::text IS NULL OR r.plan_name ILIKE '%' || $4 || '%')"

    if zlat is not None and zlon is not None:
        sql = """
        SELECT
          h.name AS hospital_name,
          h.address AS address,
          h.state AS state,
          h.zipcode AS zipcode,
          h.phone AS phone,
          h.latitude AS latitude,
          h.longitude AS longitude,
          (3959 * acos(
              cos(radians($6)) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians($7)) +
              sin(radians($6)) * sin(radians(h.latitude))
          )) AS distance_miles,
          r.code_type,
          r.code,
          r.payer_name,
          r.plan_name,
          r.standard_charge_discounted_cash AS standard_charge_cash,
          r.standard_charge_negotiated_dollar AS negotiated_dollar,
          r.estimated_amount AS estimated_amount
        FROM public.stg_hospital_rates r
        LEFT JOIN public.hospitals h ON h.id = r.hospital_id
        WHERE r.service_description ILIKE '%' || $2 || '%'
          AND """ + payer_clause + """ 
          AND """ + plan_clause + """ 
          AND h.latitude IS NOT NULL AND h.longitude IS NOT NULL
        ORDER BY distance_miles ASC NULLS LAST
        LIMIT $5
        """
        rows = await conn.fetch(sql, zipcode, q, payer_like, plan_like, limit, zlat, zlon)
        return [dict(r) for r in rows]

# ----------------------------
# Nearest-facility pricing (geo-first, ensures a facility list)
# ----------------------------
async def get_service_id(conn: asyncpg.Connection, code_type: str, code: str) -> Optional[int]:
    row = await conn.fetchrow(
        "SELECT id FROM public.services WHERE code_type = $1 AND code = $2 LIMIT 1",
        code_type,
        code,
    )
    return int(row["id"]) if row else None


async def get_service_ids(conn: asyncpg.Connection, service_query: str, code_type: Optional[str] = None, code: Optional[str] = None, limit: int = 25) -> List[int]:
    """Return a list of candidate service_ids for a user query.

    We prefer keyword matches in `services` (description/explanations), and we also include the
    explicitly-resolved (code_type, code) id when provided.
    """
    q = (service_query or "").strip()
    ids: List[int] = []

    # 1) Keyword matches
    if q:
        rows = await conn.fetch(
            """
            SELECT id
            FROM public.services
            WHERE (cpt_explanation ILIKE '%' || $1 || '%'
                OR service_description ILIKE '%' || $1 || '%'
                OR patient_summary ILIKE '%' || $1 || '%')
            ORDER BY id
            LIMIT $2
            """,
            q,
            limit,
        )
        ids.extend([int(r["id"]) for r in rows if r and r["id"] is not None])

    # 2) Explicit code match (ensures at least one id if the keyword search is sparse)
    if code_type and code:
        sid = await get_service_id(conn, code_type, code)
        if sid is not None:
            ids.append(int(sid))

    # de-dup, preserve order
    out: List[int] = []
    seen = set()
    for i in ids:
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out



async def price_lookup_nearest_facilities(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    service_query: str,
    payment_mode: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    limit: int = 10,
    radius_array: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns nearest facilities (by distance to ZIP centroid) with best-available price fields.

    For cash: pulls MIN(standard_charge_cash) (fallback to MIN(estimated_amount), MIN(standard_charge_gross))
    For insurance: pulls the first matching plan row by payer/plan filters with negotiated_dollar preferred.
    """
    radius_array = radius_array or [10, 25, 50, 100]
    z = await conn.fetchrow("SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1", zipcode)
    if not z:
        return []

    zlat, zlon = float(z["latitude"]), float(z["longitude"])
    service_ids = await get_service_ids(conn, service_query=service_query, code_type=code_type, code=code, limit=25)
    if not service_ids:
        return []

    # We expand radius until we have enough rows, then return the best set.
    last_rows: List[Dict[str, Any]] = []
    for r in radius_array:
        if payment_mode.lower() == "cash":
            rows = await conn.fetch(
                """
                WITH user_zip AS (
                    SELECT $1::double precision AS zlat, $2::double precision AS zlon
                )
                SELECT
                    h.id AS hospital_id,
                    h.name AS hospital_name,
                    h.address,
                    h.state,
                    h.zipcode,
                    h.phone,
                    (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                    )) AS distance_miles,
                    nr.best_price,
                    nr.standard_charge_cash,
                    nr.estimated_amount,
                    nr.standard_charge_gross
                FROM public.hospitals h
                JOIN LATERAL (
                    SELECT
                        COALESCE(
                            MIN(standard_charge_cash),
                            MIN(estimated_amount),
                            MIN(standard_charge_gross)
                        ) AS best_price,
                        MIN(standard_charge_cash) AS standard_charge_cash,
                        MIN(estimated_amount) AS estimated_amount,
                        MIN(standard_charge_gross) AS standard_charge_gross
                    FROM public.negotiated_rates
                    WHERE hospital_id = h.id
                      AND service_id = ANY($3::int[])
                ) nr ON TRUE
                WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
                  AND (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                  )) <= $4
                  AND nr.best_price IS NOT NULL
                ORDER BY distance_miles ASC
                LIMIT $5
                """,
                zlat, zlon, service_ids, r, limit,
            )
        else:
            payer_pat = f"%{payer_like}%" if payer_like else "%"
            plan_pat = f"%{plan_like}%" if plan_like else "%"
            rows = await conn.fetch(
                """
                WITH user_zip AS (
                    SELECT $1::double precision AS zlat, $2::double precision AS zlon
                )
                SELECT
                    h.id AS hospital_id,
                    h.name AS hospital_name,
                    h.address,
                    h.state,
                    h.zipcode,
                    h.phone,
                    (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                    )) AS distance_miles,
                    pick.best_price,
                    pick.negotiated_dollar,
                    pick.estimated_amount,
                    pick.standard_charge_cash,
                    pick.payer_name,
                    pick.plan_name
                FROM public.hospitals h
                JOIN LATERAL (
                    SELECT
                        COALESCE(nr.negotiated_dollar, nr.estimated_amount, nr.standard_charge_cash) AS best_price,
                        nr.negotiated_dollar,
                        nr.estimated_amount,
                        nr.standard_charge_cash,
                        ip.payer_name,
                        ip.plan_name
                    FROM public.negotiated_rates nr
                    JOIN public.insurance_plans ip ON ip.id = nr.plan_id
                    WHERE nr.hospital_id = h.id
                      AND nr.service_id = ANY($3::int[])
                      AND ip.payer_name ILIKE $4
                      AND ip.plan_name ILIKE $5
                      AND (nr.negotiated_dollar IS NOT NULL OR nr.estimated_amount IS NOT NULL OR nr.standard_charge_cash IS NOT NULL)
                    ORDER BY
                        nr.negotiated_dollar NULLS LAST,
                        nr.estimated_amount NULLS LAST,
                        nr.standard_charge_cash NULLS LAST
                    LIMIT 1
                ) pick ON TRUE
                WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
                  AND (3959 * acos(
                        cos(radians((SELECT zlat FROM user_zip))) * cos(radians(h.latitude)) *
                        cos(radians(h.longitude) - radians((SELECT zlon FROM user_zip))) +
                        sin(radians((SELECT zlat FROM user_zip))) * sin(radians(h.latitude))
                  )) <= $6
                ORDER BY distance_miles ASC
                LIMIT $7
                """,
                zlat, zlon, service_ids, payer_pat, plan_pat, r, limit,
            )

        last_rows = [dict(rr) for rr in rows] if rows else last_rows
        if len(last_rows) >= MIN_FACILITIES_TO_DISPLAY:
            return last_rows

    return last_rows


    # No lat/lon available, return without distance ordering
    sql2 = """
    SELECT
      COALESCE(h.name, r.hospital_name) AS hospital_name,
      h.address AS address,
      h.state AS state,
      h.zipcode AS zipcode,
      h.phone AS phone,
      NULL::float AS distance_miles,
      r.code_type,
      r.code,
      r.payer_name,
      r.plan_name,
      r.standard_charge_discounted_cash AS standard_charge_cash,
      r.standard_charge_negotiated_dollar AS negotiated_dollar,
      r.estimated_amount AS estimated_amount
    FROM public.stg_hospital_rates r
    LEFT JOIN public.hospitals h ON h.id = r.hospital_id
    WHERE r.service_description ILIKE '%' || $2 || '%'
      AND """ + payer_clause + """ 
      AND """ + plan_clause + """ 
    ORDER BY COALESCE(h.state, ''), COALESCE(h.zipcode, ''), COALESCE(h.name, r.hospital_name, '')
    LIMIT $5
    """
    rows = await conn.fetch(sql2, zipcode, q, payer_like, plan_like, limit)
    return [dict(r) for r in rows]


async def price_lookup_progressive(
    conn: asyncpg.Connection,
    zipcode: str,
    code_type: str,
    code: str,
    service_query: str,
    payer_like: Optional[str],
    plan_like: Optional[str],
    payment_mode: str,
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Geo-first nearest-facility pricing lookup with expanding radius."""
    radius_array = [10, 25, 50, 100]

    rows = await price_lookup_nearest_facilities(
        conn,
        zipcode=zipcode,
        code_type=code_type,
        code=code,
        service_query=service_query,
        payment_mode=payment_mode,
        payer_like=payer_like,
        plan_like=plan_like,
        limit=max(MIN_FACILITIES_TO_DISPLAY, 10),
        radius_array=radius_array,
    )

    used_radius = None
    if rows:
        used_radius = radius_array[-1]
    return rows, used_radius

# ----------------------------
# Nearby hospitals (for facility list even when no prices)
# ----------------------------
async def get_nearby_hospitals(conn: asyncpg.Connection, zipcode: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Returns nearby hospitals with name/address/state/zipcode/phone (+ distance if possible).
    Uses public.hospital_details for facility directory fields:
      name, address, state, zipcode, phone, latitude, longitude
    """
    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1",
        zipcode,
    )
    zlat = float(z["latitude"]) if z and z["latitude"] is not None else None
    zlon = float(z["longitude"]) if z and z["longitude"] is not None else None

    if zlat is not None and zlon is not None:
        q = """
        SELECT
          h.name AS hospital_name,
          h.address AS address,
          h.state AS state,
          h.zipcode AS zipcode,
          h.phone AS phone,
          h.latitude AS latitude,
          h.longitude AS longitude,
          (3959 * acos(
              cos(radians($2)) * cos(radians(h.latitude)) * cos(radians(h.longitude) - radians($3)) +
              sin(radians($2)) * sin(radians(h.latitude))
          )) AS distance_miles
        FROM public.hospital_details h
        WHERE h.latitude IS NOT NULL AND h.longitude IS NOT NULL
        ORDER BY distance_miles ASC
        LIMIT $1
        """
        rows = await conn.fetch(q, limit, zlat, zlon)
        return [dict(r) for r in rows]

    # If we can’t compute distance, still return facilities
    q2 = """
    SELECT
      h.name AS hospital_name,
      h.address AS address,
      h.state AS state,
      h.zipcode AS zipcode,
      h.phone AS phone,
      NULL::float AS distance_miles
    FROM public.hospital_details h
    ORDER BY h.state, h.zipcode, h.name
    LIMIT $1
    """
    rows = await conn.fetch(q2, limit)
    return [dict(r) for r in rows]


# ----------------------------
# Estimated range (when DB price missing)
# ----------------------------
def estimate_cost_range(service_query: str, payment_mode: str) -> str:
    """
    Returns a short '$X–$Y' range, explicitly an estimate.
    This is used ONLY when DB pricing is missing.
    """
    system = (
        "You output ONLY a short numeric range for a U.S. healthcare service.\n"
        "Format: '$X–$Y'. No extra text.\n"
        "Be conservative and plausible."
    )
    user = json.dumps({"service": service_query, "payment_mode": payment_mode})
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            timeout=12,
        )
        txt = (resp.choices[0].message.content or "").strip()
        # basic sanity
        if "$" in txt and any(ch.isdigit() for ch in txt):
            return txt
        return "$1,000–$3,000"
    except Exception:
        return "$1,000–$3,000"


# ----------------------------
# Facility formatting
# ----------------------------
def _pick_price_fields(row: Dict[str, Any]) -> Optional[float]:
    """
    Best-effort: try common column names returned by your SQL function.
    """
    for k in ["best_price","standard_charge_cash","standard_charge_discounted_cash","cash_price","cash","negotiated_dollar","standard_charge_negotiated_dollar","negotiated_percentage","standard_charge_negotiated_percentage","estimated_amount","standard_charge","standard_charge_gross","rate","price","allowed_amount"]:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        # sometimes Decimal
        try:
            if v is not None and str(v).replace(".", "", 1).isdigit():
                return float(v)
        except Exception:
            pass
    return None


def _pick_hospital_name(row: Dict[str, Any]) -> str:
    return (row.get("hospital_name") or row.get("name") or "Unknown facility").strip()

def _pick_phone(row: Dict[str, Any]) -> str:
    return (row.get("phone") or "").strip()


def _pick_address(row: Dict[str, Any]) -> str:
    # Build a readable one-liner from common fields
    addr = row.get("address") or row.get("street_address") or ""
    city = row.get("city") or ""
    state = row.get("state") or ""
    z = row.get("zipcode") or row.get("zip") or ""
    parts = [p for p in [addr, city, state, z] if p]
    return ", ".join(parts).strip()


def _format_money(v: Optional[float]) -> str:
    if v is None:
        return ""
    try:
        return f"${v:,.0f}"
    except Exception:
        return f"${v}"


def build_facility_block(
    service_query: str,
    payment_mode: str,
    priced_results: List[Dict[str, Any]],
    fallback_hospitals: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (text_answer, facilities_payload_for_ui).
    Always produces at least MIN_FACILITIES_TO_DISPLAY facilities in text when possible.
    """
    facilities: List[Dict[str, Any]] = []

    # 1) Start with priced results (dedupe by name)
    seen = set()
    for r in priced_results:
        name = _pick_hospital_name(r)
        key = name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        facilities.append(r)
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY:
            break

    # 2) Top up with nearby hospitals if we still need more
    for h in fallback_hospitals:
        if len(facilities) >= MIN_FACILITIES_TO_DISPLAY:
            break
        name = (h.get("hospital_name") or "").strip() or "Unknown facility"
        key = name.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        facilities.append(h)

    # If still empty, return a graceful message (should be rare if hospitals are loaded)
    if not facilities:
        return (
            "I couldn’t find hospitals near that ZIP code in my database yet. "
            "Try another nearby ZIP, or expand the search radius.",
            [],
        )

    # Prepare an estimate range if needed
    est_range = estimate_cost_range(service_query or "this service", payment_mode or "cash")

    # Build answer text
    bullets = []
    bullets.append(f"- Colonoscopies can be **screening** (preventive) or **diagnostic** (symptoms/abnormal finding).")
    bullets.append(f"- Facility setting matters: **outpatient endoscopy centers** often differ from **hospital outpatient** pricing.")
    bullets.append(f"- If a biopsy or polyp removal happens, total cost can increase.")

    lines = []
    lines.extend(bullets)
    lines.append("")
    lines.append(f"Here are nearby options for **{service_query or 'colonoscopy'}** ({payment_mode}):")

    for i, f in enumerate(facilities[:MIN_FACILITIES_TO_DISPLAY], start=1):
        name = _pick_hospital_name(f) if "hospital_name" not in f else (f.get("hospital_name") or _pick_hospital_name(f))
        addr = _pick_address(f)
        phone = _pick_phone(f) if "phone" not in f else (f.get("phone") or _pick_phone(f))
        dist = f.get("distance_miles") or f.get("distance") or f.get("miles")
        dist_txt = ""
        try:
            if dist is not None:
                dist_txt = f" ({float(dist):.1f} mi)"
        except Exception:
            pass

        price = _pick_price_fields(f)
        if price is not None:
            price_txt = _format_money(price)
            price_note = "DB price"
        else:
            price_txt = est_range
            price_note = "ESTIMATE (no DB price yet)"

        detail_bits = []
        if addr:
            detail_bits.append(addr)
        if phone:
            detail_bits.append(f"Tel: {phone}")
        if detail_bits:
            detail = " | ".join(detail_bits)
        else:
            detail = "Contact info not available in DB."

        lines.append(f"{i}) **{name}**{dist_txt}")
        lines.append(f"   - {detail}")
        lines.append(f"   - Price: **{price_txt}** ({price_note})")

    lines.append("")
    lines.append("Confirm with the facility and your insurer.")

    # Build facility payload for UI
    ui_payload = []
    for f in facilities[:MIN_FACILITIES_TO_DISPLAY]:
        ui_payload.append(
            {
                "hospital_name": f.get("hospital_name") or _pick_hospital_name(f),
                "address": _pick_address(f),
                "phone": f.get("phone") or _pick_phone(f),
                "distance_miles": f.get("distance_miles") or f.get("distance"),
                "price": _pick_price_fields(f),
                "estimated_range": None if _pick_price_fields(f) is not None else est_range,
                "price_is_estimate": _pick_price_fields(f) is None,
            }
        )

    return "\n".join(lines), ui_payload


# ----------------------------
# Service refiners (DB-first, Python fallback)
# ----------------------------
_refiners_cache: Optional[dict] = None
_refiners_cache_loaded_at: float = 0.0


def _norm(s: str) -> str:
    return (s or "").strip().lower()


async def load_refiners_from_db(conn: asyncpg.Connection) -> dict:
    rows = await conn.fetch(
        """
        SELECT id, title, keywords, require_choice_before_pricing,
               preview_code_type, preview_code, question_text
        FROM public.service_refiner
        WHERE is_active = true
        ORDER BY id
        """
    )
    if not rows:
        return {"version": 1, "refiners": []}

    ids = [r["id"] for r in rows]
    crows = await conn.fetch(
        """
        SELECT refiner_id, choice_key, choice_label, code_type, code, sort_order
        FROM public.service_refiner_choice
        WHERE is_active = true AND refiner_id = ANY($1::text[])
        ORDER BY refiner_id, sort_order, choice_key
        """,
        ids,
    )

    choices_by_ref: Dict[str, List[dict]] = {}
    for c in crows:
        choices_by_ref.setdefault(c["refiner_id"], []).append(
            {"key": c["choice_key"], "label": c["choice_label"], "code_type": c["code_type"], "code": c["code"]}
        )

    refiners: List[dict] = []
    for r in rows:
        preview = None
        if r["preview_code_type"] and r["preview_code"]:
            preview = {"code_type": r["preview_code_type"], "code": r["preview_code"]}
        refiners.append(
            {
                "id": r["id"],
                "title": r["title"],
                "match": {"keywords": list(r["keywords"] or [])},
                "require_choice_before_pricing": bool(r["require_choice_before_pricing"]),
                "preview_code": preview,
                "question": r["question_text"],
                "choices": choices_by_ref.get(r["id"], []),
            }
        )

    return {"version": 1, "refiners": refiners}


async def get_refiners(conn: asyncpg.Connection) -> dict:
    global _refiners_cache, _refiners_cache_loaded_at
    now = time.time()
    if _refiners_cache and (now - _refiners_cache_loaded_at) < REFINERS_CACHE_TTL_SECONDS:
        return _refiners_cache

    data = None
    try:
        data = await load_refiners_from_db(conn)
    except Exception as e:
        logger.warning(f"Refiners DB load failed, using Python fallback: {e}")

    if not data or not data.get("refiners"):
        data = refiners_registry()

    _refiners_cache = data
    _refiners_cache_loaded_at = now
    return data


def match_refiner(service_query: str, refiners_doc: dict) -> Optional[dict]:
    q = _norm(service_query)
    if not q:
        return None
    for ref in refiners_doc.get("refiners", []):
        kws = [_norm(k) for k in (ref.get("match", {}).get("keywords") or [])]
        if any(k and k in q for k in kws):
            return ref
    return None


def apply_refiner_choice(message: str, merged: dict, refiner: Optional[dict]) -> dict:
    if not refiner:
        return merged
    choice_key = (message or "").strip()
    if not choice_key:
        return merged
    for ch in refiner.get("choices", []):
        if str(ch.get("key")) == choice_key:
            merged["code_type"] = ch.get("code_type")
            merged["code"] = ch.get("code")
            merged["refiner_id"] = refiner.get("id")
            merged["refiner_choice"] = choice_key
            return merged
    return merged


def maybe_apply_preview_code(merged: dict, refiner: Optional[dict]) -> dict:
    if not refiner:
        return merged
    if refiner.get("require_choice_before_pricing") is True:
        return merged
    if merged.get("code_type") and merged.get("code"):
        return merged
    preview = refiner.get("preview_code")
    if preview and preview.get("code_type") and preview.get("code"):
        merged["code_type"] = preview["code_type"]
        merged["code"] = preview["code"]
        merged["refiner_id"] = refiner.get("id")
    return merged


def needs_refinement(merged: dict, refiner: Optional[dict]) -> bool:
    if not refiner:
        return False
    if merged.get("refiner_choice"):
        return False

    code_type = merged.get("code_type")
    code = merged.get("code")
    preview = refiner.get("preview_code")

    if refiner.get("require_choice_before_pricing") is True:
        return True
    if not (code_type and code):
        return True
    if preview and code_type == preview.get("code_type") and code == preview.get("code"):
        return True
    return False


def get_refinement_prompt(refiner: dict) -> str:
    lines = [refiner.get("question", "").strip(), ""]
    for ch in refiner.get("choices", []):
        lines.append(f"{ch.get('key')}) {ch.get('label')}")
    lines.append("")
    lines.append("Reply with the number that fits best.")
    return "\n".join([l for l in lines if l])


# ----------------------------
# Main streaming endpoint
# ----------------------------
@app.post("/chat_stream")
async def chat_stream(req: ChatRequest, request: Request):
    require_auth(request)
    ip = request.client.host if request.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(429, detail="Rate limit exceeded.")
    if not pool:
        raise HTTPException(500, detail="Database not ready")

    async def event_gen():
        try:
            async with pool.acquire() as conn:
                session_id, state = await get_or_create_session(conn, req.session_id)
                yield sse({"type": "session", "session_id": session_id})
                await save_message(conn, session_id, "user", req.message)

                # 1) Intent + state merge
                intent = await extract_intent(req.message, state)
                merged = merge_state(state, intent)

                # If this message is a NEW pricing question (service inferred) and the user did not provide a ZIP,
                # do not reuse a prior ZIP/payment from the session. Force ZIP collection first.
                if intent.get("_new_price_question"):
                    if not message_contains_zip(req.message):
                        merged.pop("zipcode", None)
                    # Always reset payment/payer/plan for a new pricing question unless explicitly stated
                    if not message_contains_payment_info(req.message):
                        merged.pop("payment_mode", None)
                        merged.pop("payer_like", None)
                        merged.pop("plan_like", None)
                        merged.pop("cash_only", None)
                    merged.pop("_awaiting", None)
                _normalize_payment_mode(merged)

                # 2) Force/override pricing intent when we are mid-flow
                intent = apply_intent_override_if_needed(intent, req.message, merged, session_id)
                mode = intent.get("mode") or "hybrid"

                # 3) OPTIONAL: reset gating fields only when this message looks like a NEW pricing question
                # (Never reset while we are awaiting ZIP/payment/payer)
                if (
                    RESET_GATING_FIELDS_ON_NEW_PRICE_QUESTION
                    and mode in ["price", "hybrid"]
                    and not (merged.get("_awaiting") in {"zip", "payment", "payer"})
                ):
                    # Only reset if the message itself contains a price keyword AND a service hint
                    msg_l = (req.message or "").lower()
                    inferred_service = infer_service_query_from_message(req.message)
                    if inferred_service and any(kw in msg_l for kw in INTENT_OVERRIDE_FORCE_PRICE_KEYWORDS):
                        reset_gating_fields_for_new_price_question(req.message, merged)

                # 4) Load refiners + apply choice (if user replies 1/2/3)
                refiners_doc = await get_refiners(conn)
                refiner = match_refiner(merged.get("service_query") or "", refiners_doc)
                merged = apply_refiner_choice(req.message, merged, refiner)
                merged = maybe_apply_preview_code(merged, refiner)

                # ----------------------------
                # GENERAL mode
                # ----------------------------
                if mode == "general":
                    system = "You are CostSavvy.health. Answer clearly in plain language. Avoid medical advice."
                    parts: List[str] = []
                    for chunk in stream_llm_to_sse(system, req.message, parts):
                        yield chunk
                    full_answer = "".join(parts).strip() or "I couldn’t generate a response."
                    await save_message(conn, session_id, "assistant", full_answer, {"mode": "general", "intent": intent})
                    await update_session_state(conn, session_id, merged)
                    await log_query(conn, session_id, req.message, intent, None, 0, False, full_answer)
                    yield sse({"type": "final", "used_web_search": False})
                    return

                # ----------------------------
                # PRICE / HYBRID mode (deterministic, DB-first)
                # ----------------------------
                if mode in ["price", "hybrid"]:

                    # Universal Variant-First Gate (DB)
                    # Do not collect ZIP/payment or price until a CPT-backed variant is confirmed.
                    # 1) Handle pending confirmation
                    if merged.get("_awaiting") == "variant_confirm" and isinstance(merged.get("_variant_confirm_choice"), dict):
                        ans = (req.message or "").strip().lower()
                        if ans in {"yes", "y", "yeah", "yep", "correct", "right", "ok", "okay"}:
                            choice = merged["_variant_confirm_choice"]
                            merged["code_type"] = "CPT"
                            merged["code"] = choice.get("cpt_code")
                            merged["variant_id"] = choice.get("id")
                            merged["service_query"] = choice.get("variant_name") or merged.get("service_query")
                            merged.pop("_awaiting", None)
                            merged.pop("_variant_confirm_choice", None)
                        elif ans in {"no", "n", "nope", "incorrect", "wrong"}:
                            merged["_awaiting"] = "service_variant_text"
                            merged.pop("_variant_confirm_choice", None)
                            msg = "Got it. Tell me the specific service in a few words (include body part, with/without contrast, views, or screening vs diagnostic if relevant)."
                            yield sse({"type": "delta", "text": msg})
                            await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant"})
                            await update_session_state(conn, session_id, merged)
                            yield sse({"type": "final", "used_web_search": False})
                            return

                    # 2) Handle numbered selection
                    if merged.get("_awaiting") == "service_variant" and isinstance(merged.get("_variant_choices"), list):
                        merged = apply_service_variant_choice(req.message, merged, merged.get("_variant_choices") or [])

                    # 3) Handle free-text refinement (we'll search again using this)
                    if merged.get("_awaiting") == "service_variant_text":
                        free = (req.message or "").strip()
                        if free:
                            merged["variant_free_text"] = free
                            merged.pop("_awaiting", None)

                    # 4) If we still don't have a CPT, search service_variants using the user text and gate here.
                    if not (merged.get("code_type") and merged.get("code")) and merged.get("variant_id") is None:
                        query_text = (merged.get("variant_free_text") or merged.get("service_query") or req.message or "").strip()
                        candidates = await get_service_variants_by_text(conn, query_text, limit=15)

                        seen = set()
                        uniq = []
                        for c in candidates:
                            key = (str(c.get("cpt_code") or ""), str(c.get("variant_name") or ""))
                            if key in seen:
                                continue
                            seen.add(key)
                            uniq.append(c)

                        if len(uniq) == 1:
                            merged["_awaiting"] = "variant_confirm"
                            merged["_variant_confirm_choice"] = {
                                "id": uniq[0].get("id"),
                                "cpt_code": uniq[0].get("cpt_code"),
                                "variant_name": uniq[0].get("variant_name"),
                            }
                            msg = build_variant_confirm_prompt(uniq[0])
                            yield sse({"type": "delta", "text": msg})
                            await save_message(conn, session_id, "assistant", msg, {"mode": "confirm_variant"})
                            await update_session_state(conn, session_id, merged)
                            yield sse({"type": "final", "used_web_search": False})
                            return

                        if len(uniq) > 1:
                            merged["_awaiting"] = "service_variant"
                            merged["_variant_choices"] = [
                                {
                                    "id": u.get("id"),
                                    "parent_service": u.get("parent_service"),
                                    "cpt_code": u.get("cpt_code"),
                                    "variant_name": u.get("variant_name"),
                                    "patient_summary": u.get("patient_summary"),
                                    "is_preventive": u.get("is_preventive"),
                                    "cpt_explanation": u.get("cpt_explanation"),
                                }
                                for u in uniq[:10]
                            ]
                            label = (merged.get("service_query") or "this service")
                            msg = build_variant_options_prompt(label, merged["_variant_choices"])
                            yield sse({"type": "delta", "text": msg})
                            await save_message(conn, session_id, "assistant", msg, {"mode": "choose_variant"})
                            await update_session_state(conn, session_id, merged)
                            yield sse({"type": "final", "used_web_search": False})
                            return

                        merged["_awaiting"] = "service_variant_text"
                        msg = (
                            "I can price this, but I need a bit more detail on the exact service. "
                            "For example: body part, with/without contrast, number of views, screening vs diagnostic, or the test name your clinician ordered."
                        )
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant"})
                        await update_session_state(conn, session_id, merged)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # Gate 0: need a service (or a code)
                    if not (merged.get("service_query") or "").strip() and not (merged.get("code_type") and merged.get("code")):
                        # Try deterministic inference for common services
                        inferred = infer_service_query_from_message(req.message)
                        if inferred:
                            merged["service_query"] = inferred

                    # Gate 1: ZIP required
                    zipcode = merged.get("zipcode")
                    if not zipcode:
                        merged["_awaiting"] = "zip"
                        msg = "What’s your 5-digit ZIP code?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_zip", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return
                    if merged.get("_awaiting") == "zip":
                        merged.pop("_awaiting", None)

                    # Gate 2: payment mode required
                    payment_mode = merged.get("payment_mode")
                    if not payment_mode:
                        merged["_awaiting"] = "payment"
                        msg = "Are you paying cash (self-pay) or using insurance? If insurance, what carrier (e.g., Aetna)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_payment", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return
                    if merged.get("_awaiting") == "payment":
                        merged.pop("_awaiting", None)

                    _normalize_payment_mode(merged)
                    payment_mode = merged.get("payment_mode") or payment_mode

                    # Gate 3 (insurance): require payer
                    if payment_mode == "insurance" and not (merged.get("payer_like") or "").strip():
                        merged["_awaiting"] = "payer"
                        msg = "Which insurance carrier should I match prices for (e.g., Aetna, UnitedHealthcare, Blue Cross)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_payer", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return
                    if merged.get("_awaiting") == "payer" and (merged.get("payer_like") or "").strip():
                        merged.pop("_awaiting", None)

                    # Refiners: ask for choice before pricing when required
                    if refiner and refiner.get("require_choice_before_pricing") is True and not merged.get("refiner_choice"):
                        msg = get_refinement_prompt(refiner)
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_variant", "refiner_id": refiner.get("id"), "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # Resolve code if needed
                    if not (merged.get("code_type") and merged.get("code")):
                        resolved = await resolve_service_code(conn, merged)
                        if resolved:
                            merged["code_type"], merged["code"] = resolved

                    # If still no service, ask
                    if not (merged.get("service_query") or "").strip() and not (merged.get("code_type") and merged.get("code")):
                        msg = "What service are you pricing (for example: colonoscopy, MRI brain, chest x-ray, office visit, lab test)?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # If code is still missing, ask for more detail
                    if not (merged.get("code_type") and merged.get("code")):
                        msg = "I can price this, but I need a bit more detail on the exact service. What exactly is being ordered?"
                        yield sse({"type": "delta", "text": msg})
                        await save_message(conn, session_id, "assistant", msg, {"mode": "clarify_service_detail", "intent": intent})
                        await update_session_state(conn, session_id, merged)
                        await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                        yield sse({"type": "final", "used_web_search": False})
                        return

                    # ---- PRICED LOOKUP (geo-first, progressive radius) ----
                    results, used_max_radius = await price_lookup_progressive(
                        conn,
                        merged["zipcode"],
                        merged["code_type"],
                        merged["code"],
                        merged.get("service_query") or "",
                        merged.get("payer_like"),
                        merged.get("plan_like"),
                        merged.get("payment_mode") or "cash",
                    )

                    # Always fetch at least 5 facilities for display
                    try:
                        nearby_hospitals = await get_nearby_hospitals(conn, merged["zipcode"], limit=MIN_FACILITIES_TO_DISPLAY)
                    except Exception as e:
                        logger.warning(f"Nearby hospitals lookup failed: {e}")
                        nearby_hospitals = []

                    facility_text, facility_payload = build_facility_block(
                        service_query=merged.get("service_query") or "this service",
                        payment_mode=merged.get("payment_mode") or "cash",
                        priced_results=results,
                        fallback_hospitals=nearby_hospitals,
                    )

                    yield sse(
                        {
                            "type": "results",
                            "results": results[:25],
                            "facilities": facility_payload,
                            "state": {
                                k: merged.get(k)
                                for k in [
                                    "zipcode",
                                    "payment_mode",
                                    "payer_like",
                                    "plan_like",
                                    "service_query",
                                    "code_type",
                                    "code",
                                    "refiner_id",
                                    "refiner_choice",
                                ]
                            },
                        }
                    )
                    yield sse({"type": "delta", "text": facility_text})

                    full_answer = facility_text
                    await save_message(
                        conn,
                        session_id,
                        "assistant",
                        full_answer,
                        {
                            "mode": mode,
                            "intent": intent,
                            "result_count": len(results),
                            "used_max_radius": used_max_radius,
                            "refiner_id": (refiner or {}).get("id"),
                        },
                    )
                    await update_session_state(conn, session_id, merged)

                    used_web = len(results) < MIN_DB_RESULTS_BEFORE_WEB
                    await log_query(conn, session_id, req.message, intent, used_max_radius, len(results), used_web, full_answer)
                    yield sse({"type": "final", "used_web_search": used_web})
                    return

                # Fallback: ask to rephrase (should be rare)
                msg = "I’m not sure what you need. Can you rephrase?"
                yield sse({"type": "delta", "text": msg})
                await save_message(conn, session_id, "assistant", msg, {"mode": "fallback", "intent": intent})
                await update_session_state(conn, session_id, merged)
                await log_query(conn, session_id, req.message, intent, None, 0, False, msg)
                yield sse({"type": "final", "used_web_search": False})

        except Exception as e:
            logger.exception("Unhandled error")
            yield sse({"type": "error", "message": f"{type(e).__name__}: {str(e)}"})
            yield sse({"type": "final", "used_web_search": False})

    headers = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@app.post("/chat")

async def chat(_req: ChatRequest, _request: Request):
    raise HTTPException(410, detail="Use /chat_stream")