import json
import os
import re
import urllib.request
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(title="Invoice extractor")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExtractRequest(BaseModel):
    text: str = Field(max_length=100_000)

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class Invoice(BaseModel):
    vendor: str = Field(min_length=1)
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


# ------------------------------------------------------------------ amounts
SYMBOL_TO_CODE = {"US$": "USD", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
WORD_TO_CODE = {"dollar": "USD", "dollars": "USD", "euro": "EUR", "euros": "EUR",
                "pound": "GBP", "pounds": "GBP"}
CODES = "USD|EUR|GBP|INR|JPY|CAD|AUD|CHF|CNY|SGD|NZD|AED|SEK|NOK|DKK|ZAR|BRL|MXN"
SYM = r"US\$|\$|€|£|¥|₹"
NUM = r"\d{1,3}(?:\.\d{3})+,\d{2}|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d{1,2})?"

PRE_RE = re.compile(rf"(?P<cur>{CODES}|{SYM})\s?(?P<num>{NUM})(?!\d)")
POST_RE = re.compile(rf"(?<![\d.,])(?P<num>{NUM})\s?(?P<cur>{CODES}|dollars?|euros?|pounds?)\b", re.I)
KW_NUM_RE = re.compile(rf"(?:total|amount|balance|due|payable|pay)[^\d\n]{{0,25}}(?P<num>{NUM})(?!\d)", re.I)

AMOUNT_POS = [("total due", 9), ("amount due", 9), ("balance due", 9), ("amount payable", 9),
              ("total payable", 9), ("grand total", 8), ("total amount", 8), ("net payable", 8),
              ("invoice total", 8), ("please pay", 7), ("owed", 6), ("total", 6), ("remit", 5),
              ("due", 5), ("payable", 5), ("pay", 4), ("balance", 4), ("amount", 3)]
AMOUNT_NEG = [("subtotal", -8), ("sub-total", -8), ("sub total", -8), ("tax", -6), ("vat", -6),
              ("gst", -6), ("discount", -6), ("shipping", -5), ("deposit", -5), ("paid", -5),
              ("fee", -3), ("tip", -3)]
TAX_INCL_RE = re.compile(r"\b(?:incl(?:uding|uded|\.)?|inc\.?|with|plus|excl(?:uding|\.)?|before|after)\s+(?:tax|vat|gst)\b", re.I)


def parse_number(s: str) -> float:
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+,\d{2}", s):
        s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s) else s.replace(",", ".")
    return float(s)


def to_code(tok: str) -> str:
    t = tok.strip()
    if t in SYMBOL_TO_CODE:
        return SYMBOL_TO_CODE[t]
    if t.lower() in WORD_TO_CODE:
        return WORD_TO_CODE[t.lower()]
    return t.upper()


def nearest_keyword(segment: str, table) -> int:
    segment = segment.lower()
    best = None
    for kw, weight in table:
        idx = segment.rfind(kw)
        if idx >= 0:
            key = (idx + len(kw), len(kw))
            if best is None or key > best[0]:
                best = (key, weight)
    return best[1] if best else 0


def segment_before(text: str, pos: int, split_re: str) -> str:
    before = text[max(0, pos - 90):pos]
    return re.split(split_re, before)[-1]


def extract_amount(text: str):
    cands = {}
    for m in PRE_RE.finditer(text):
        cands[m.start("num")] = (m.start(), m.group("num"), to_code(m.group("cur")))
    for m in POST_RE.finditer(text):
        cands.setdefault(m.start("num"), (m.start(), m.group("num"), to_code(m.group("cur"))))

    scored = []
    for _, (start, num, cur) in cands.items():
        try:
            value = parse_number(num)
        except ValueError:
            continue
        seg = TAX_INCL_RE.sub(" ", segment_before(text, start, r"[\n;|]|,\s"))
        scored.append((nearest_keyword(seg, AMOUNT_POS + AMOUNT_NEG), value, start, cur))
    if scored:
        scored.sort(key=lambda c: (c[0], c[1]))
        score, value, _, cur = scored[-1]
        return value, cur

    # no currency next to a number: fall back to keyword + number
    best = None
    for m in KW_NUM_RE.finditer(text):
        if re.match(r"\d{4}[-/]\d", text[m.start("num"):m.start("num") + 8]):
            continue
        try:
            value = parse_number(m.group("num"))
        except ValueError:
            continue
        score = nearest_keyword(text[m.start():m.start("num")], AMOUNT_POS + AMOUNT_NEG)
        if best is None or (score, value) > best[:2]:
            best = (score, value)
    if best:
        return best[1], None
    return None, None


def any_currency(text: str) -> Optional[str]:
    m = re.search(rf"\b({CODES})\b", text)
    if m:
        return m.group(1)
    m = re.search(SYM, text)
    if m:
        return to_code(m.group(0))
    m = re.search(r"\b(dollars?|euros?|pounds?)\b", text, re.I)
    return to_code(m.group(1)) if m else None


# -------------------------------------------------------------------- dates
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
         r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)")
DATE_PATTERNS = [
    ("iso", re.compile(r"(?<!\d)(?P<y>\d{4})[-/.](?P<mo>\d{1,2})[-/.](?P<d>\d{1,2})(?!\d)")),
    ("mdy", re.compile(rf"\b(?P<m>{MONTH})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b", re.I)),
    ("dmy", re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{MONTH})\.?,?\s+(?P<y>\d{{4}})\b", re.I)),
    ("num", re.compile(r"(?<![\d./-])(?P<a>\d{1,2})[/.-](?P<b>\d{1,2})[/.-](?P<y>\d{4})(?!\d)")),
]
DATE_POS = [("due date", 10), ("due", 9), ("payable by", 9), ("pay by", 9), ("pay on or before", 9),
            ("payment by", 8), ("deadline", 8), ("settle by", 8), ("before", 5), ("payment date", 4)]
DATE_NEG = [("invoice date", -7), ("issued", -7), ("issue date", -7), ("date of issue", -7),
            ("dated", -6), ("created", -6), ("order date", -6), ("billing date", -6),
            ("invoiced", -7), ("sent", -4)]


def _mk_date(y, mo, d) -> Optional[str]:
    try:
        return datetime(int(y), int(mo), int(d)).strftime("%Y-%m-%d")
    except ValueError:
        return None


def find_dates(text: str):
    found = []
    for kind, rx in DATE_PATTERNS:
        for m in rx.finditer(text):
            if kind == "iso":
                iso = _mk_date(m["y"], m["mo"], m["d"])
            elif kind in ("mdy", "dmy"):
                iso = _mk_date(m["y"], MONTHS[m["m"][:3].lower()], m["d"])
            else:
                a, b = int(m["a"]), int(m["b"])
                iso = _mk_date(m["y"], b, a) if a > 12 else _mk_date(m["y"], a, b)
            if iso:
                found.append((m.start(), m.end(), iso))
    found.sort()
    result, last_end = [], -1
    for s, e, iso in found:
        if s >= last_end:
            result.append((s, e, iso))
            last_end = e
    return result


def extract_date(text: str) -> Optional[str]:
    dates = find_dates(text)
    if not dates:
        return None
    scored = []
    for i, (s, e, iso) in enumerate(dates):
        seg = segment_before(text, s, r"[\n;|]")
        score = nearest_keyword(seg, DATE_POS + DATE_NEG)
        if score <= 0 and "due" in text[e:e + 25].lower():
            score = 6
        scored.append((score, i, iso))
    scored.sort()
    return scored[-1][2]


# ------------------------------------------------------------------- vendor
SUFFIXES = {"ltd", "limited", "inc", "incorporated", "llc", "llp", "gmbh", "corp", "corporation",
            "co", "company", "industries", "group", "holdings", "solutions", "systems",
            "technologies", "labs", "enterprises", "partners", "services", "trading", "plc",
            "ag", "sa", "pty", "associates", "brothers", "bros", "international", "global",
            "supplies", "works", "consulting", "studio", "studios", "ventures", "logistics"}
STOP_WORDS = {"for", "total", "amount", "due", "invoice", "dated", "please", "payment", "plus",
              "regarding", "re", "bill", "billed", "owes", "charges", "is", "has", "was", "on", "to"}
LABEL_RE = re.compile(
    r"(?i:\b(?:vendor(?:\s+name)?|supplier|seller|payee|biller|merchant|company|billed\s+by|"
    r"bill\s+from|issued\s+by|invoice\s+from|invoiced\s+by|pay\s+to|remit(?:tance)?\s+to|"
    r"payable\s+to|sold\s+by|from))\s*[:\-–]?\s*(?P<rest>[^\n;|]+)")


def _norm(tok: str) -> str:
    return tok.strip(".,;:()").lower()


def _chain_end(tokens, i):
    j = i
    while j + 1 < len(tokens) and _norm(tokens[j + 1]) in SUFFIXES:
        j += 1
    return j


def _clean(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().rstrip(",;:")


def vendor_from_label(text: str):
    fallback = None
    for m in LABEL_RE.finditer(text):
        rest = m.group("rest").strip()
        if not rest or not (rest[0].isupper() or rest[0].isdigit()):
            continue
        tokens = rest.split()
        hit = next((i for i, t in enumerate(tokens) if _norm(t) in SUFFIXES), None)
        if hit is not None:
            return _clean(" ".join(tokens[: _chain_end(tokens, hit) + 1]))
        kept = []
        for t in tokens[:6]:
            if t in STOP_WORDS:
                break
            kept.append(t)
            if t.endswith((",", ":")):
                break
        if kept and fallback is None:
            fallback = _clean(" ".join(kept))
    return fallback


def vendor_by_suffix(text: str):
    tokens = text.split()
    for i, t in enumerate(tokens):
        if _norm(t) not in SUFFIXES:
            continue
        j = _chain_end(tokens, i)
        k = i
        while k > 0 and i - k < 5:
            prev = tokens[k - 1]
            if prev.endswith((",", ":", ";")) or not (prev[0].isupper() or (prev[0].isdigit() and re.search(r"[A-Za-z]", prev))):
                break
            k -= 1
        if k < i or j > i:
            return _clean(" ".join(tokens[k: j + 1]))
    return None


def extract_vendor(text: str) -> Optional[str]:
    labeled = vendor_from_label(text)
    if labeled and any(_norm(t) in SUFFIXES for t in labeled.split()):
        return labeled
    return vendor_by_suffix(text) or labeled


def heuristic_extract(text: str) -> dict:
    amount, cur = extract_amount(text)
    return {
        "vendor": extract_vendor(text),
        "amount": amount,
        "currency": cur or any_currency(text),
        "date": extract_date(text),
    }


# ----------------------------------------------- optional local-LLM fallback
def llm_extract(text: str) -> dict:
    """Used only when OLLAMA_URL is set (e.g. a tunnel to a local Ollama) and a field is missing."""
    base = os.getenv("OLLAMA_URL", "").rstrip("/")
    if not base:
        return {}
    schema = {
        "type": "object",
        "properties": {"vendor": {"type": "string"}, "amount": {"type": "number"},
                       "currency": {"type": "string"}, "date": {"type": "string"}},
        "required": ["vendor", "amount", "currency", "date"],
    }
    payload = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2:1b"),
        "stream": False,
        "format": schema,
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content":
                      "Extract vendor, total amount due (number), 3-letter uppercase currency code, "
                      "and payment due date (YYYY-MM-DD) from this invoice. Reply with JSON only.\n\n" + text}],
    }
    try:
        req = urllib.request.Request(base + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            content = json.loads(r.read())["message"]["content"]
        data = json.loads(content)
    except Exception:
        return {}
    out = {}
    if isinstance(data.get("vendor"), str) and data["vendor"].strip():
        out["vendor"] = data["vendor"].strip()
    if isinstance(data.get("amount"), (int, float)) and not isinstance(data["amount"], bool):
        out["amount"] = float(data["amount"])
    if isinstance(data.get("currency"), str) and re.fullmatch(r"[A-Za-z]{3}", data["currency"].strip()):
        out["currency"] = data["currency"].strip().upper()
    if isinstance(data.get("date"), str) and re.search(r"\d{4}-\d{2}-\d{2}", data["date"]):
        out["date"] = re.search(r"\d{4}-\d{2}-\d{2}", data["date"]).group(0)
    return out


# ----------------------------------------------------------------- endpoint
@app.post("/extract", response_model=Invoice)
async def extract(req: ExtractRequest):
    try:
        fields = heuristic_extract(req.text)
        if any(v in (None, "") for v in fields.values()):
            for k, v in llm_extract(req.text).items():
                if fields.get(k) in (None, ""):
                    fields[k] = v
        if any(v in (None, "") for v in fields.values()):
            raise ValueError("incomplete")
        return Invoice(**fields)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not extract invoice fields from the text")


@app.get("/")
async def root():
    return {"service": "invoice extractor", "endpoint": "POST /extract"}
