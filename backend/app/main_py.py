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

# Always try to show at least this many facilities in the answer
MIN_FACILITIES_TO_DISPLAY = int(os.getenv("MIN_FACILITIES_TO_DISPLAY", "5"))

# Intent override controls
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
    """Streams OpenAI chat.completions deltas as SSE 'delta' events."""
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
"""


def merge_state(state: Dict[str, Any], intent: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(state or {})
    for k in ["zipcode", "radius_miles", "payer_like", "plan_like", "payment_mode", "service_query", "code_type", "code", "cash_only"]:
        v = intent.get(k)
        if v is not None and v != "":
            out[k] = v
    return out


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
    """Deterministic session-aware intent extraction."""
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
        for k, v in carrier_map.items():
            if k in ml:
                return v

        clean = m
        for stop in ["i have", "i use", "use", "with", "insurance", "my", "have", "paying"]:
            clean = re.sub(r'\b' + re.escape(stop) + r'\b', '', clean, flags=re.IGNORECASE)

        clean = clean.strip()
        tokens = re.findall(r"[a-zA-Z]+", clean)

        if len(tokens) == 1 and len(tokens[0]) >= 3:
            return tokens[0].title()
        
        if 2 <= len(tokens) <= 3:
            return " ".join([t.title() for t in tokens])
            
        return None

    # 1) If awaiting ZIP, keep asking
    if awaiting == "zip":
        return {"mode": "clarify", "clarifying_question": "What's your 5-digit ZIP code?"}

    # 2) If awaiting payment, accept cash/insurance
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

    # 3) Session-aware continuation
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

    # 4) New price question without ZIP
    inferred_service = infer_service_query_from_message(msg)
    if inferred_service and not have_zip:
        return {"mode": "price", "service_query": inferred_service, "clarifying_question": None, "_new_price_question": True}

    # 5) LLM fallback
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
                extra={"session_id": session_id, "prev_mode": prev},
            )
        intent["mode"] = "price"
        intent["intent_overridden"] = True

        if not (merged.get("service_query") or "").strip():
            inferred = infer_service_query_from_message(message)
            if inferred:
                merged["service_query"] = inferred
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
# Service Variants (CORRECTED FOR ACTUAL SCHEMA)
# ----------------------------
async def get_service_variants_by_text(conn: asyncpg.Connection, user_text: str, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Search service_variants table for matching CPT codes.
    ACTUAL SCHEMA: cpt_code (int), cpt_explanation (text), patient_summary (text), category (varchar)
    """
    text = (user_text or "").strip()
    if not text:
        return []

    tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", text.lower()) if len(t) >= 3][:8]
    full_pat = f"%{text.lower()}%"
    pats = [f"%{t}%" for t in tokens]

    params: List[Any] = [full_pat] + pats
    
    # Build WHERE clauses
    ors = ["lower(cpt_explanation) LIKE $1", "lower(patient_summary) LIKE $1"]
    for i in range(2, len(params) + 1):
        ors.append(f"lower(cpt_explanation) LIKE ${i}")
        ors.append(f"lower(patient_summary) LIKE ${i}")

    where_sql = " OR ".join(ors) if ors else "FALSE"

    # Build scoring
    score_sql = "CASE WHEN (lower(cpt_explanation) LIKE $1 OR lower(patient_summary) LIKE $1) THEN 100 ELSE 0 END"
    for i in range(2, len(params) + 1):
        score_sql += f" + CASE WHEN (lower(cpt_explanation) LIKE ${i} OR lower(patient_summary) LIKE ${i}) THEN 5 ELSE 0 END"

    rows = await conn.fetch(
        f"""
        SELECT
          cpt_code,
          cpt_explanation,
          patient_summary,
          category,
          ({score_sql}) AS score
        FROM public.service_variants
        WHERE {where_sql}
        ORDER BY score DESC, cpt_code ASC
        LIMIT {int(limit)}
        """,
        *params,
    )
    return [dict(r) for r in rows]


def _simplify_variant_for_patient(v: Dict[str, Any]) -> str:
    """Convert technical CPT to patient-friendly description."""
    expl = (v.get("cpt_explanation") or "").strip()
    summary = (v.get("patient_summary") or "").strip()

    base = summary or expl
    base = re.sub(r"\s+", " ", base).strip()
    if len(base) > 160:
        base = base[:157].rstrip() + "..."

    low = base.lower()
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
    """Build numbered list of variant options."""
    label = (service_label or "this service").strip()
    lines = [f"Before I look up prices, which specific **{label.upper()}** do you mean?", ""]
    
    for i, v in enumerate(variants, start=1):
        # Use cpt_explanation as the variant name
        vname = (v.get("cpt_explanation") or f"CPT {v.get('cpt_code')}").strip()
        desc = _simplify_variant_for_patient(v)
        lines.append(f"{i}) {vname} — {desc}")
    
    lines.append("")
    lines.append("Reply with the number that best matches what you need.")
    return "\n".join(lines)


def build_variant_confirm_prompt(v: Dict[str, Any]) -> str:
    """Build confirmation prompt for single match."""
    vname = (v.get("cpt_explanation") or f"CPT {v.get('cpt_code')}").strip()
    return f"Just to confirm, you mean: **{vname}**. Is that right? (yes/no)"


def apply_service_variant_choice(message: str, merged: dict, choices: List[dict]) -> dict:
    """Handle numbered user selection (1, 2, 3, etc.)."""
    choice_str = (message or "").strip()
    if not choice_str:
        return merged
    
    try:
        choice_num = int(choice_str)
        if 1 <= choice_num <= len(choices):
            chosen = choices[choice_num - 1]
            merged["code_type"] = "CPT"
            merged["code"] = str(chosen.get("cpt_code"))
            merged["cpt_code"] = chosen.get("cpt_code")
            merged["service_query"] = chosen.get("cpt_explanation") or merged.get("service_query")
            merged.pop("_awaiting", None)
            merged.pop("_variant_choices", None)
    except (ValueError, IndexError):
        pass
    
    return merged


# ----------------------------
# Price lookup (CORRECTED FOR ACTUAL SCHEMA)
# ----------------------------
async def price_lookup_from_hospital_details(
    conn: asyncpg.Connection,
    zipcode: str,
    code: str,
    payment_mode: str,
    payer_like: Optional[str],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """
    Query hospital_details table for pricing.
    Schema: hospital_id, hospital_name, service_description, code_Type, code,
            standard_charge_gross, standard_charge_discounted_cash, payer_name, plan_name,
            standard_charge_negotiated_dollar, estimated_amount, etc.
    """
    # Get ZIP coordinates
    z = await conn.fetchrow(
        "SELECT latitude, longitude FROM public.zip_locations WHERE zipcode = $1 LIMIT 1",
        zipcode,
    )
    
    if not z or z["latitude"] is None or z["longitude"] is None:
        # No coordinates, just return results without distance
        if payment_mode == "cash":
            where_clause = "code = $1 AND (standard_charge_discounted_cash IS NOT NULL OR estimated_amount IS NOT NULL)"
            params = [code, limit]
        else:
            where_clause = "code = $1 AND payer_name ILIKE $2 AND standard_charge_negotiated_dollar IS NOT NULL"
            params = [code, f"%{payer_like or ''}%", limit]
        
        rows = await conn.fetch(
            f"""
            SELECT
                hospital_name,
                service_description,
                code,
                standard_charge_discounted_cash AS cash_price,
                standard_charge_negotiated_dollar AS negotiated_dollar,
                estimated_amount,
                payer_name,
                plan_name
            FROM public.hospital_details
            WHERE {where_clause}
            ORDER BY hospital_name
            LIMIT ${len(params)}
            """,
            *params,
        )
        return [dict(r) for r in rows]
    
    zlat, zlon = float(z["latitude"]), float(z["longitude"])
    
    # With coordinates, calculate distance
    if payment_mode == "cash":
        sql = """
        SELECT
            hospital_name,
            service_description,
            code,
            standard_charge_discounted_cash AS cash_price,
            estimated_amount,
            (3959 * acos(
                cos(radians($2)) * cos(radians(latitude)) * 
                cos(radians(longitude) - radians($3)) +
                sin(radians($2)) * sin(radians(latitude))
            )) AS distance_miles
        FROM public.hospital_details
        WHERE code = $1 
          AND (standard_charge_discounted_cash IS NOT NULL OR estimated_amount IS NOT NULL)
        ORDER BY distance_miles ASC
        LIMIT $4
        """
        params = [code, zlat, zlon, limit]
    else:
        sql = """
        SELECT
            hospital_name,
            service_description,
            code,
            standard_charge_negotiated_dollar AS negotiated_dollar,
            estimated_amount,
            payer_name,
            plan_name,
            (3959 * acos(
                cos(radians($2)) * cos(radians(latitude)) * 
                cos(radians(longitude) - radians($3)) +
                sin(radians($2)) * sin(radians(latitude))
            )) AS distance_miles
        FROM public.hospital_details
        WHERE code = $1 
          AND payer_name ILIKE $4
          AND standard_charge_negotiated_dollar IS NOT NULL
        ORDER BY distance_miles ASC
        LIMIT $5
        """
        params = [code, zlat, zlon, f"%{payer_like or ''}%", limit]
    
    rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]


async def price_lookup_from_stg_hospital_rates(
    conn: asyncpg.Connection,
    zipcode: str,
    code: str,
    payment_mode: str,
    payer_like: Optional[str],
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """
    Fallback to stg_hospital_rates table.
    Schema has: hospital_id, service_description, code, standard_charge_discounted_cash,
                standard_charge_negotiated_dollar, payer_name, plan_name, estimated_amount, etc.
    """
    if payment_mode == "cash":
        where_clause = "code = $1 AND (standard_charge_discounted_cash IS NOT NULL OR estimated_amount IS NOT NULL)"
        params = [code, limit]
    else:
        where_clause = "code = $1 AND payer_name ILIKE $2"
        params = [code, f"%{payer_like or ''}%", limit]
    
    rows = await conn.fetch(
        f"""
        SELECT
            hospital_id,
            service_description,
            code,
            standard_charge_discounted_cash AS cash_price,
            standard_charge_negotiated_dollar AS negotiated_dollar,
            estimated_amount,
            payer_name,
            plan_name
        FROM public.stg_hospital_rates
        WHERE {where_clause}
        ORDER BY hospital_id
        LIMIT ${len(params)}
        """,
        *params,
    )
    return [dict(r) for