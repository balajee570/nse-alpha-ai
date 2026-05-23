from __future__ import annotations
import re
import io
import os
import json
import uuid
import time
import math
import pickle
import logging
import warnings
import threading
import tempfile
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tavily import TavilyClient

# Silence Streamlit warnings from the background thread that has no ScriptRunContext.
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*to view a Streamlit app.*")
logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> datetime:
    return datetime.now(tz=IST)

def fmt_ist(fmt: str = "%d %b %Y · %H:%M IST") -> str:
    return now_ist().strftime(fmt)

# Yahoo Finance recently tightened crumb / Cloudflare checks, which causes
# Ticker.info / .financials / .balance_sheet to fail with HTTP 401 "Invalid
# Crumb" using the default requests session. curl_cffi impersonates a real
# Chrome TLS fingerprint and bypasses the block.
try:
    from curl_cffi import requests as _cf_requests
    YF_SESSION = _cf_requests.Session(impersonate="chrome")
except Exception:
    YF_SESSION = None

def _yf_ticker(symbol: str):
    """Return a yf.Ticker bound to the impersonating session when available."""
    if YF_SESSION is not None:
        try:
            return yf.Ticker(symbol, session=YF_SESSION)
        except Exception:
            pass
    return yf.Ticker(symbol)

def _yf_download(*args, **kwargs):
    """yf.download wrapper that passes the curl_cffi session when supported."""
    if YF_SESSION is not None and "session" not in kwargs:
        try:
            return yf.download(*args, session=YF_SESSION, **kwargs)
        except TypeError:
            # yfinance version doesn't accept session=; fall through
            pass
    return yf.download(*args, **kwargs)

# =============================
# BACKGROUND JOB RUNNER
# Lets the heavy pipeline (scan + AI + fundamentals) keep running even if
# the user closes the tab. Status + result are persisted on disk so any
# session that re-opens the page reattaches to the in-flight job.
# Streamlit Cloud / `streamlit run` keeps the parent process alive long
# enough for daemon=False threads to finish; on serverless runtimes that
# scale to zero this WILL NOT survive.
# =============================

JOB_DIR     = Path(os.environ.get("NSE_JOB_DIR", "/tmp/nse_alpha_jobs"))
JOB_STATUS  = JOB_DIR / "status.json"
JOB_RESULT  = JOB_DIR / "result.pkl"
JOB_LOCK    = JOB_DIR / "job.lock"
JOB_HEARTBEAT_TIMEOUT = 600   # seconds; lock considered stale if updated_at older than this
JOB_DIR.mkdir(parents=True, exist_ok=True)

_AI_DEBUG_BUFFER: list = []   # filled by call_ai when session_state is unreachable

def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try: os.remove(tmp)
        except Exception: pass
        raise

def _write_status(d: dict) -> None:
    try:
        d = {**d, "updated_at": now_ist().isoformat()}
        _atomic_write(JOB_STATUS, json.dumps(d, default=str).encode("utf-8"))
    except Exception:
        pass

def _read_status() -> dict | None:
    try:
        if not JOB_STATUS.exists():
            return None
        return json.loads(JOB_STATUS.read_text(encoding="utf-8"))
    except Exception:
        return None

def _acquire_lock(job_id: str) -> None:
    JOB_LOCK.write_text(json.dumps({"job_id": job_id, "started_at": now_ist().isoformat()}))

def _release_lock() -> None:
    try:
        if JOB_LOCK.exists():
            JOB_LOCK.unlink()
    except Exception:
        pass

def _lock_age_seconds() -> float | None:
    try:
        lock = json.loads(JOB_LOCK.read_text(encoding="utf-8"))
        started = datetime.fromisoformat(lock.get("started_at", ""))
        if started.tzinfo is None:
            started = started.replace(tzinfo=IST)
        return (now_ist() - started).total_seconds()
    except Exception:
        return None

def _job_alive() -> bool:
    """True iff a lock exists AND (no status yet but lock is fresh, OR status updated within HEARTBEAT_TIMEOUT and not done)."""
    if not JOB_LOCK.exists():
        return False
    status = _read_status()

    # If status exists and signals completion, clear the lock and report not-alive.
    if status and status.get("done"):
        _release_lock()
        return False

    # No status file yet — give the thread up to 30 s to write its first heartbeat.
    if not status:
        age = _lock_age_seconds()
        if age is None or age > 30:
            _release_lock()
            return False
        return True

    # Status present and not done: check heartbeat freshness.
    try:
        last = datetime.fromisoformat(status.get("updated_at", ""))
        if last.tzinfo is None:
            last = last.replace(tzinfo=IST)
        age = (now_ist() - last).total_seconds()
        if age > JOB_HEARTBEAT_TIMEOUT:
            _release_lock()
            return False
    except Exception:
        return False
    return True

def _save_result(payload: dict) -> None:
    _atomic_write(JOB_RESULT, pickle.dumps(payload))

def _load_result() -> dict | None:
    try:
        if not JOB_RESULT.exists():
            return None
        return pickle.loads(JOB_RESULT.read_bytes())
    except Exception:
        return None

def _result_mtime() -> float:
    try:
        return JOB_RESULT.stat().st_mtime if JOB_RESULT.exists() else 0.0
    except Exception:
        return 0.0

def _make_status_cb(job_id: str):
    """Factory for a progress callback that writes JOB_STATUS atomically.
    Signature: cb(phase: str, current: int, total: int, msg: str = "", done: bool = False)
    """
    started_at = now_ist().isoformat()
    def cb(phase: str, current: int = 0, total: int = 100, msg: str = "", done: bool = False, error: str | None = None):
        _write_status({
            "job_id": job_id,
            "phase":  phase,
            "current": int(current) if current is not None else 0,
            "total":   int(total) if total is not None else 0,
            "msg":     msg,
            "started_at": started_at,
            "done":    bool(done),
            "error":   error,
        })
    return cb

def start_background_job() -> str:
    """Spawn the background pipeline. Returns the new job_id.
    Acquires the lock and writes the first status entry synchronously so the
    UI can see _job_alive() == True immediately on the next rerun."""
    job_id = uuid.uuid4().hex[:8]
    _acquire_lock(job_id)
    cb = _make_status_cb(job_id)
    cb("startup", 0, 100, "Starting background scan")
    t = threading.Thread(
        target=_run_pipeline_bg, args=(job_id, cb),
        daemon=False, name=f"nse-job-{job_id}",
    )
    t.start()
    return job_id

def _run_pipeline_bg(job_id: str, cb) -> None:
    try:
        result = _run_pipeline(cb)
        _save_result(result)
        cb("complete", 100, 100, "Scan complete", done=True)
    except Exception as e:
        cb("error", 0, 100, f"Job failed: {e}", done=True, error=str(e))
    finally:
        _release_lock()

# Forward declaration — defined later once all pipeline functions exist.
_run_pipeline = None  # type: ignore

# =============================
# API CONFIG
# =============================

SARVAM_API_KEY = st.secrets["SARVAM_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]
SARVAM_URL     = "https://api.sarvam.ai/v1/chat/completions"

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# =============================
# SCANNER CONSTANTS
# =============================

MA_PERIOD          = 50
MA_LONG_PERIOD     = 200
BREAKOUT_LOOKBACK  = 20
VOL_LOOKBACK       = 20
VOL_SURGE_THRESH   = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS           = 60
CHUNK_SIZE         = 80
MAX_DISPLAY_STOCKS = 12
RSI_PERIOD         = 14
RSI_OVERBOUGHT     = 72   # stocks above this are excluded from Top 3 picks

# ── Rally classifier / shortlist ────────────────────────────────────
SHORTLIST_TARGET     = 25
SHORTLIST_MIN_SCORE  = 55.0
SHORTLIST_FLOOR      = 10
FUND_WORKERS         = 8
FUND_CACHE_TTL_SEC   = 21600   # 6 h

STAGE_LABELS = {
    0: "—",
    1: "Accumulation",
    2: "Early Markup",
    3: "Breakout",
    4: "Extended",
}
STAGE_WEIGHTS = {0: 0.20, 1: 0.85, 2: 1.00, 3: 0.75, 4: 0.15}

# =============================
# ETF REGISTRY — fetched live from NSE, zero hardcoding
# Tries the NSE ETF CSV endpoint; yfinance symbol = SYMBOL + ".NS"
# =============================

@st.cache_data(ttl=86400)
def get_etf_registry() -> pd.DataFrame:
    """
    Download NSE ETF list. Aggressively filter to only genuine ETF symbols:
    - No leading digits
    - Alphanumeric only (no special chars)
    - Symbol length 3–20 chars
    - Series == 'EQ' if column present
    - Must contain ETF-like keyword in symbol OR name
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept"    : "text/html,application/xhtml+xml,*/*;q=0.8",
        "Referer"   : "https://www.nseindia.com/",
    }
    candidates = [
        "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv",
        "https://nsearchives.nseindia.com/content/ETF/eq_etfseclist.csv",
        "https://nsearchives.nseindia.com/content/equities/etf_seclist.csv",
    ]

    # Keywords that appear in genuine NSE ETF symbols or names
    ETF_KEYWORDS = [
        "ETF","BEES","GOLD","SILVER","NIFTY","SENSEX","BANK","IT","PHARMA",
        "INFRA","CPSE","BHARAT","DEFENCE","MIDCAP","SMALLCAP","CONSUMPTION",
        "AUTO","METAL","REALTY","MEDIA","ENERGY","LIQUID","GILT","BOND","DEBT",
    ]

    for url in candidates:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = df.columns.str.strip().str.upper()

            sym_col  = next((c for c in df.columns if "SYMBOL" in c), None)
            name_col = next((c for c in df.columns if any(k in c for k in ["NAME","SECURITY","SCRIP","FUND"])), None)
            ser_col  = next((c for c in df.columns if "SERIES" in c), None)
            if sym_col is None:
                continue

            df = df.rename(columns={sym_col: "SYMBOL"})
            df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
            df["NAME"]   = df[name_col].astype(str).str.strip() if name_col else df["SYMBOL"]

            # ── Filter 1: Series must be EQ if column exists
            if ser_col:
                df = df[df[ser_col].astype(str).str.strip().str.upper() == "EQ"]

            # ── Filter 2: Symbol must be clean alphanumeric, 3–20 chars, no leading digit
            df = df[df["SYMBOL"].str.match(r'^[A-Z][A-Z0-9]{2,19}$', na=False)]

            # ── Filter 3: Symbol or Name must contain an ETF keyword
            kw_pattern = "|".join(ETF_KEYWORDS)
            mask = (
                df["SYMBOL"].str.contains(kw_pattern, case=False, na=False) |
                df["NAME"].str.contains(kw_pattern, case=False, na=False)
            )
            df = df[mask]

            if df.empty:
                continue

            df["YF_SYMBOL"] = df["SYMBOL"] + ".NS"
            return df[["SYMBOL","NAME","YF_SYMBOL"]].drop_duplicates("SYMBOL").reset_index(drop=True)

        except Exception:
            continue

    return pd.DataFrame()

SECTOR_MAP = {
    "HDFCBANK":"Banking","ICICIBANK":"Banking","SBIN":"Banking","AXISBANK":"Banking",
    "KOTAKBANK":"Banking","INDUSINDBK":"Banking","BANDHANBNK":"Banking","FEDERALBNK":"Banking",
    "IDFCFIRSTB":"Banking","PNB":"Banking","BANKBARODA":"Banking","CANBK":"Banking",
    "BAJFINANCE":"Finance","BAJAJFINSV":"Finance","CHOLAFIN":"Finance","MUTHOOTFIN":"Finance",
    "HDFC":"Finance","RECLTD":"Finance","PFC":"Finance","IRFC":"Finance",
    "TCS":"IT","INFY":"IT","WIPRO":"IT","HCLTECH":"IT","TECHM":"IT",
    "LTIM":"IT","MPHASIS":"IT","PERSISTENT":"IT","COFORGE":"IT","OFSS":"IT",
    "SUNPHARMA":"Pharma","DRREDDY":"Pharma","CIPLA":"Pharma","DIVISLAB":"Pharma",
    "APOLLOHOSP":"Healthcare","MAXHEALTH":"Healthcare","FORTIS":"Healthcare",
    "LUPIN":"Pharma","TORNTPHARM":"Pharma","AUROPHARMA":"Pharma","ALKEM":"Pharma",
    "HEROMOTOCO":"Auto","EICHERMOT":"Auto","TVSMOTOR":"Auto","ASHOKLEY":"Auto",
    "MOTHERSON":"Auto","BOSCHLTD":"Auto","BHARATFORG":"Auto","BALKRISIND":"Auto",
    "TATASTEEL":"Metals","JSWSTEEL":"Metals","HINDALCO":"Metals","VEDL":"Metals",
    "SAIL":"Metals","NMDC":"Metals","COALINDIA":"Metals","NATIONALUM":"Metals",
    "RELIANCE":"Energy","ONGC":"Energy","IOC":"Energy","BPCL":"Energy",
    "GAIL":"Energy","POWERGRID":"Energy","NTPC":"Energy","ADANIGREEN":"Energy",
    "TATAPOWER":"Energy","TORNTPOWER":"Energy","CESC":"Energy",
    "HINDUNILVR":"FMCG","ITC":"FMCG","NESTLEIND":"FMCG","BRITANNIA":"FMCG",
    "DABUR":"FMCG","MARICO":"FMCG","COLPAL":"FMCG","EMAMILTD":"FMCG",
    "LT":"Infra","ADANIPORTS":"Infra","SIEMENS":"CapGoods","ABB":"CapGoods",
    "BHEL":"CapGoods","HAL":"Defence","BEL":"Defence","COCHINSHIP":"Defence",
    "DLF":"Realty","GODREJPROP":"Realty","OBEROIRLTY":"Realty","PHOENIXLTD":"Realty",
    "PRESTIGE":"Realty","BRIGADE":"Realty",
    "BHARTIARTL":"Telecom","IDEA":"Telecom","INDUSTOWER":"Telecom",
    "PIDILITIND":"Chemicals","SRF":"Chemicals","DEEPAKNITR":"Chemicals",
    "NAVINFLUOR":"Chemicals","ATUL":"Chemicals",
}

# =============================
# PAGE CONFIG
# =============================

st.set_page_config(page_title="NSE Alpha AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg-base:     #080c18;
    --gold:        #d4a843;
    --gold-light:  #f0c96a;
    --gold-dim:    rgba(212,168,67,0.18);
    --gold-border: rgba(212,168,67,0.28);
    --blue:        #4f8ef7;
    --emerald:     #00c896;
    --amber:       #f59e0b;
    --red:         #f87171;
    --text-1:      #eef2ff;
    --text-2:      #8896b3;
    --text-3:      #4a5578;
    --border:      rgba(255,255,255,0.07);
}

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; background:var(--bg-base); color:var(--text-1); }

.stApp {
    background:
        radial-gradient(ellipse 90% 55% at 15% -5%,  rgba(79,142,247,0.08) 0%, transparent 55%),
        radial-gradient(ellipse 70% 45% at 85% 105%, rgba(212,168,67,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 50% 30% at 50% 50%,  rgba(0,200,150,0.03) 0%, transparent 60%),
        #080c18;
}

.nse-header { padding:36px 0 24px; }
.nse-logo-row { display:flex; align-items:baseline; gap:14px; margin-bottom:10px; }
.nse-wordmark {
    font-family:'Cinzel',serif; font-weight:900; font-size:2.9rem; letter-spacing:0.07em;
    background:linear-gradient(110deg,#b8873a 0%,#d4a843 25%,#f0c96a 50%,#d4a843 75%,#b8873a 100%);
    background-size:250% auto; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; animation:goldShimmer 5s linear infinite; line-height:1;
}
@keyframes goldShimmer{0%{background-position:0% center}100%{background-position:250% center}}
.nse-badge {
    font-family:'JetBrains Mono',monospace; font-size:0.58rem; font-weight:700;
    letter-spacing:0.22em; color:#080c18;
    background:linear-gradient(135deg,var(--gold),var(--gold-light));
    padding:3px 10px; border-radius:3px; text-transform:uppercase;
    vertical-align:middle; position:relative; top:-4px;
}
.nse-tagline { font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--text-3); letter-spacing:0.18em; text-transform:uppercase; }
.nse-tagline em { color:var(--text-2); font-style:normal; }
.header-divider { height:1px; margin-top:20px; background:linear-gradient(90deg,transparent,var(--gold-dim) 20%,var(--gold-border) 50%,var(--gold-dim) 80%,transparent); }

.section-header { display:flex; align-items:center; gap:12px; margin:34px 0 16px; font-family:'Cinzel',serif; font-size:0.62rem; font-weight:700; letter-spacing:0.3em; text-transform:uppercase; color:var(--gold); }
.section-header::before { content:''; width:3px; height:13px; background:linear-gradient(180deg,var(--gold-light),var(--gold)); border-radius:2px; flex-shrink:0; box-shadow:0 0 8px var(--gold-dim); }
.section-header::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,var(--gold-border),transparent); }

.pick-card { background:rgba(15,20,40,0.7); border:1px solid var(--border); border-top:1px solid rgba(212,168,67,0.2); border-radius:14px; padding:22px 24px; backdrop-filter:blur(16px); position:relative; overflow:hidden; transition:border-color 0.3s,box-shadow 0.3s,transform 0.2s; margin-bottom:4px; }
.pick-card::before { content:''; position:absolute; top:0; left:0; right:0; height:1px; background:linear-gradient(90deg,transparent,rgba(212,168,67,0.5),transparent); }
.pick-card:hover { border-color:var(--gold-border); box-shadow:0 16px 48px rgba(212,168,67,0.07); transform:translateY(-3px); }
.ticker-name { font-family:'JetBrains Mono',monospace; font-size:1.3rem; font-weight:700; color:var(--text-1); letter-spacing:0.05em; }
.ticker-meta { font-size:0.78rem; color:var(--text-3); margin:5px 0 14px; }
.upside-pill { display:inline-flex; align-items:center; gap:4px; background:linear-gradient(135deg,var(--gold),var(--gold-light)); color:#1a1000; font-weight:700; font-size:0.74rem; padding:4px 13px; border-radius:99px; font-family:'JetBrains Mono',monospace; letter-spacing:0.05em; }
.target-row { display:flex; align-items:center; gap:8px; margin-top:13px; font-size:0.8rem; color:var(--text-3); }
.target-price { font-family:'JetBrains Mono',monospace; font-weight:700; font-size:1.05rem; color:var(--emerald); }
.signal-tag { margin-top:8px; font-family:'JetBrains Mono',monospace; font-size:0.66rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; }
.sig-breakout { color:var(--emerald); }
.sig-building  { color:var(--amber); }

.strategy-wrap { background:rgba(4,14,10,0.75); border:1px solid rgba(0,200,150,0.1); border-left:2px solid var(--emerald); border-radius:0 14px 14px 0; padding:26px 30px; backdrop-filter:blur(12px); }
.strategy-wrap p  { font-size:0.87rem; line-height:1.85; color:#c0e8d8; margin:0 0 10px; }
.strategy-wrap h1,.strategy-wrap h2,.strategy-wrap h3,.strategy-wrap h4 { font-family:'Cinzel',serif; color:var(--gold-light); font-size:0.76rem; letter-spacing:0.12em; text-transform:uppercase; margin:20px 0 8px; font-weight:700; }
.strategy-wrap ul  { padding-left:18px; margin:4px 0 12px; }
.strategy-wrap li  { font-size:0.85rem; line-height:1.8; color:#c0e8d8; margin-bottom:3px; }
.strategy-wrap strong { color:var(--gold-light); font-weight:600; }
.strategy-wrap table { width:100%; border-collapse:collapse; margin:10px 0 16px; font-size:0.82rem; }
.strategy-wrap th { font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.14em; text-transform:uppercase; color:var(--gold); padding:8px 12px; border-bottom:1px solid var(--gold-border); background:rgba(212,168,67,0.05); text-align:left; }
.strategy-wrap td { padding:8px 12px; color:var(--text-2); border-bottom:1px solid rgba(255,255,255,0.04); vertical-align:top; line-height:1.6; }
.strategy-wrap tr:last-child td { border-bottom:none; }
.strategy-wrap tr:hover td { background:rgba(255,255,255,0.02); }

.intel-wrap { background:rgba(8,14,30,0.75); border:1px solid var(--border); border-left:2px solid var(--blue); border-radius:0 12px 12px 0; padding:22px 24px; backdrop-filter:blur(10px); }
.intel-wrap p  { font-size:0.84rem; line-height:1.82; color:var(--text-2); margin:0 0 10px; }
.intel-wrap h1,.intel-wrap h2,.intel-wrap h3,.intel-wrap h4 { font-family:'JetBrains Mono',monospace; color:var(--blue); font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; margin:18px 0 8px; font-weight:700; border-bottom:1px solid rgba(79,142,247,0.15); padding-bottom:6px; }
.intel-wrap ul  { padding-left:16px; margin:4px 0 12px; }
.intel-wrap li  { font-size:0.83rem; line-height:1.78; color:var(--text-2); margin-bottom:4px; }
.intel-wrap strong { color:var(--text-1); font-weight:600; }
.intel-wrap em    { color:var(--gold-light); font-style:normal; }
.intel-scan-time { font-family:'JetBrains Mono',monospace; font-size:0.6rem; font-weight:700; letter-spacing:0.2em; text-transform:uppercase; color:var(--blue); margin-bottom:14px; display:flex; align-items:center; gap:8px; }
.intel-scan-time::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,rgba(79,142,247,0.3),transparent); }

.stButton > button { background:linear-gradient(110deg,#b8873a 0%,#d4a843 35%,#f0c96a 65%,#d4a843 100%); background-size:250% auto; color:#1a1000; font-family:'Cinzel',serif; font-weight:700; font-size:0.85rem; letter-spacing:0.14em; text-transform:uppercase; border:none; padding:16px 40px; border-radius:8px; width:100%; transition:background-position 0.5s,box-shadow 0.3s,transform 0.2s; box-shadow:0 4px 20px rgba(212,168,67,0.22); }
.stButton > button:hover { background-position:right center; box-shadow:0 6px 36px rgba(212,168,67,0.38); transform:translateY(-2px); }
.stButton > button:active { transform:translateY(0); }

.stProgress > div > div { background:linear-gradient(90deg,var(--gold),var(--gold-light),var(--emerald)); background-size:200% auto; border-radius:4px; animation:progShimmer 1.8s linear infinite; }
@keyframes progShimmer{0%{background-position:0% center}100%{background-position:200% center}}

div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; border:1px solid var(--border) !important; }

[data-testid="metric-container"] { background:rgba(12,16,32,0.7) !important; border:1px solid var(--border) !important; border-top:1px solid rgba(212,168,67,0.2) !important; border-radius:12px !important; padding:18px 22px !important; backdrop-filter:blur(12px) !important; }
[data-testid="metric-container"] label { font-family:'JetBrains Mono',monospace !important; font-size:0.62rem !important; letter-spacing:0.16em !important; text-transform:uppercase !important; color:var(--text-3) !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family:'Cinzel',serif !important; font-size:1.55rem !important; font-weight:700 !important; color:var(--text-1) !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-family:'JetBrains Mono',monospace !important; font-size:0.72rem !important; }

.stCaption p { color:var(--text-3) !important; font-family:'JetBrains Mono',monospace !important; font-size:0.68rem !important; }
hr { border:none !important; height:1px !important; background:linear-gradient(90deg,transparent,var(--border),transparent) !important; }
.nse-footer { text-align:right; color:var(--text-3); font-family:'JetBrains Mono',monospace; font-size:0.66rem; margin-top:24px; padding-bottom:24px; letter-spacing:0.1em; }

.empty-state { text-align:center; padding:110px 20px 60px; }
.empty-glyph { font-family:'Cinzel',serif; font-size:4rem; color:rgba(212,168,67,0.15); margin-bottom:24px; letter-spacing:0.3em; animation:glyphPulse 4s ease-in-out infinite; }
@keyframes glyphPulse{0%,100%{opacity:0.5}50%{opacity:1}}
.empty-title { font-size:1.05rem; color:var(--text-2); margin-bottom:10px; }
.empty-sub { font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--text-3); letter-spacing:0.08em; line-height:2; }

.stSpinner > div { border-top-color:var(--gold) !important; }
[data-testid="stSidebar"] { display:none; }
</style>
""", unsafe_allow_html=True)

# =============================
# HEADER
# =============================

st.markdown(f"""
<div class="nse-header">
    <div class="nse-logo-row">
        <span class="nse-wordmark">NSE Alpha</span>
        <span class="nse-badge">AI</span>
    </div>
    <div class="nse-tagline">
        Market Intelligence &nbsp;·&nbsp;
        <em>{now_ist().strftime("%d %b %Y")}</em> &nbsp;·&nbsp;
        <em>{now_ist().strftime("%H:%M IST")}</em> &nbsp;·&nbsp;
        <em>Rally Detector · Fundamentals · Deep Dive</em>
    </div>
    <div class="header-divider"></div>
</div>
""", unsafe_allow_html=True)

# =============================
# AI CALL
# =============================

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _extract_user_answer(reasoning: str, max_chars: int = 3000) -> str:
    """When the model truncates before writing `content`, salvage the user-facing
    portion from `reasoning_content`. Strategy: find the LAST markdown section
    header (## or **bold**) and return everything from there to the end. If no
    such anchor exists, return the trailing slice."""
    if not reasoning:
        return ""
    text = _strip_think(reasoning).strip()
    if not text:
        return ""
    matches = list(re.finditer(r"(?m)^\s*(?:##+\s|\*\*[^\n*]{2,}\*\*)", text))
    if matches:
        return text[matches[-1].start():].strip()
    return text[-max_chars:].strip()

def _record_ai_debug(entry: dict) -> None:
    """Append AI call debug data to session_state, falling back to a module
    buffer if there's no script_run_ctx (i.e. background thread)."""
    try:
        log = st.session_state.get("ai_debug", [])
        log.append(entry)
        st.session_state["ai_debug"] = log
    except Exception:
        _AI_DEBUG_BUFFER.append(entry)

def call_ai(prompt: str, system: str = "", max_tokens: int = 4096) -> str:
    """Call Sarvam chat completion using sarvam-105b.

    sarvam-105b is a reasoning model: chain-of-thought goes into
    `message.reasoning_content` and the user-facing answer into
    `message.content`. On the starter subscription tier max_tokens is
    capped at 4096, so the model often spends its full budget reasoning
    and emits `content = null`. In that case we salvage the user-facing
    portion from `reasoning_content` directly so the UI is never empty
    for a successful 200 response.

    If a 400 "exceeds the maximum allowed" comes back, the per-tier cap
    is parsed from the error body and the call is retried once.
    """
    headers  = {"Authorization": f"Bearer {SARVAM_API_KEY}", "Content-Type": "application/json"}
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    model = "sarvam-105b"
    payload = {
        "model"      : model,
        "messages"   : messages,
        "max_tokens" : max_tokens,
        "temperature": 0.5,
        "top_p"      : 1,
    }

    def _post(p):
        return requests.post(SARVAM_URL, headers=headers, json=p, timeout=240)

    try:
        r = _post(payload)
        # Auto-cap retry: starter tier returns
        # "max_tokens (N) exceeds the maximum allowed for sarvam-105b for your
        #  subscription tier (starter): M"
        if r.status_code == 400 and "exceeds the maximum allowed" in r.text:
            mtch = re.search(r"subscription tier \([^)]+\):\s*(\d+)", r.text)
            if mtch:
                tier_cap = int(mtch.group(1))
                payload["max_tokens"] = min(max_tokens, tier_cap)
                _record_ai_debug({"model": model, "status": 400,
                                  "note": f"max_tokens capped to tier limit {tier_cap}",
                                  "body": r.text[:600]})
                r = _post(payload)

        debug = {"model": model, "status": r.status_code, "body": r.text[:1200]}

        if r.status_code == 200:
            choices = r.json().get("choices", [])
            if choices:
                ch        = choices[0] or {}
                msg       = ch.get("message", {}) or {}
                content   = str(msg.get("content") or "")
                reasoning = str(msg.get("reasoning_content") or "")
                debug.update({
                    "finish_reason": ch.get("finish_reason"),
                    "content_len":   len(content),
                    "reasoning_len": len(reasoning),
                })

                stripped = _strip_think(content)
                if stripped:
                    debug["fallback_used"] = False
                    _record_ai_debug(debug)
                    return stripped

                # Reasoning-content salvage: when sarvam-105b exhausts the
                # 4096-token tier cap on CoT before writing content, extract
                # the user-facing portion (last markdown section) from the
                # trace and return it cleanly.
                salvaged = _extract_user_answer(reasoning)
                if salvaged:
                    debug["fallback_used"] = True
                    _record_ai_debug(debug)
                    return salvaged

        _record_ai_debug(debug)
    except requests.exceptions.Timeout:
        _record_ai_debug({"model": model, "error": "Timed out after 240s"})
    except Exception as e:
        _record_ai_debug({"model": model, "error": str(e)})

    return ""

# =============================
# NSE REGISTRY
# =============================

def _safe_st_error(msg: str) -> None:
    try: st.error(msg)
    except Exception: print(f"[ERROR] {msg}")

@st.cache_data(ttl=86400)
def get_nse_registry() -> pd.DataFrame:
    url     = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        df.columns = df.columns.str.strip().str.upper()
        missing = {"SERIES", "SYMBOL"} - set(df.columns)
        if missing:
            _safe_st_error(f"NSE format changed. Missing: {missing}")
            return pd.DataFrame()
        df = df[df["SERIES"] == "EQ"].copy()
        df["YF_SYMBOL"] = df["SYMBOL"].astype(str).str.strip() + ".NS"
        return df.reset_index(drop=True)
    except Exception as e:
        _safe_st_error(f"NSE registry download failed: {e}")
        return pd.DataFrame()

# =============================
# TECHNICAL HELPERS
# =============================

def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 1) if not rsi.empty else np.nan

def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    high, low, cp = df["High"], df["Low"], df["Close"].shift(1)
    tr  = pd.concat([(high - low), (high - cp).abs(), (low - cp).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    price = df["Close"].iloc[-1]
    return round(float(atr / price * 100), 2) if price else np.nan

# =============================
# EXPANDED TECHNICAL HELPERS
# =============================

def _safe_last(series: pd.Series) -> float:
    try:
        v = float(series.dropna().iloc[-1])
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def compute_ma(series: pd.Series, period: int) -> float:
    if len(series) < period:
        return np.nan
    return _safe_last(series.rolling(period).mean())

def compute_ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Return (macd_last, signal_last, hist_last, hist_series)."""
    if len(series) < slow + signal:
        return np.nan, np.nan, np.nan, pd.Series(dtype=float)
    ema_fast = compute_ema_series(series, fast)
    ema_slow = compute_ema_series(series, slow)
    macd     = ema_fast - ema_slow
    sig      = compute_ema_series(macd, signal)
    hist     = macd - sig
    return _safe_last(macd), _safe_last(sig), _safe_last(hist), hist

def compute_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """Wilder ADX. Returns (adx_last, plus_di_last, minus_di_last)."""
    if len(df) < period * 2 + 1:
        return np.nan, np.nan, np.nan
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr        = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing via ewm with alpha=1/period
    atr_w     = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx       = dx.ewm(alpha=1/period, adjust=False).mean()
    return _safe_last(adx), _safe_last(plus_di), _safe_last(minus_di)

def compute_adx_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    if len(df) < period * 2 + 1:
        return pd.Series(dtype=float)
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm   = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm  = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr        = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_w     = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di   = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di  = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()

def compute_roc(series: pd.Series, period: int = 20) -> float:
    if len(series) < period + 1:
        return np.nan
    try:
        ref = float(series.iloc[-period - 1])
        if ref == 0:
            return np.nan
        return round((float(series.iloc[-1]) / ref - 1) * 100, 2)
    except Exception:
        return np.nan

def compute_obv_slope(close: pd.Series, volume: pd.Series, window: int = 20) -> float:
    if len(close) < window + 5:
        return np.nan
    obv = (np.sign(close.diff().fillna(0)) * volume.fillna(0)).cumsum()
    tail = obv.iloc[-window:]
    if tail.std() == 0 or len(tail) < window:
        return 0.0
    x = np.arange(len(tail))
    try:
        slope, _ = np.polyfit(x, tail.values, 1)
        # Normalise by mean OBV magnitude
        denom = max(abs(tail.mean()), 1.0)
        return round(float(slope) / denom, 6)
    except Exception:
        return np.nan

def compute_rs_vs_nifty(close: pd.Series, nifty_close: pd.Series, period: int = 63) -> float:
    if nifty_close is None or len(nifty_close) < period + 1 or len(close) < period + 1:
        return np.nan
    try:
        stock_ret  = float(close.iloc[-1])  / float(close.iloc[-period - 1])
        nifty_ret  = float(nifty_close.iloc[-1]) / float(nifty_close.iloc[-period - 1])
        if nifty_ret == 0:
            return np.nan
        return round(stock_ret / nifty_ret, 3)
    except Exception:
        return np.nan

def compute_bb_width_series(series: pd.Series, period: int = 20, k: float = 2.0) -> pd.Series:
    if len(series) < period:
        return pd.Series(dtype=float)
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + k * std
    lower = mid - k * std
    return (upper - lower) / mid.replace(0, np.nan)

def is_bb_squeeze(series: pd.Series, period: int = 20, k: float = 2.0, lookback: int = 120, pct: float = 0.25) -> bool:
    bbw = compute_bb_width_series(series, period, k).dropna()
    if len(bbw) < lookback:
        return False
    tail = bbw.iloc[-lookback:]
    last = float(tail.iloc[-1])
    threshold = float(tail.quantile(pct))
    return last <= threshold

def slope_pct(series: pd.Series, window: int = 20) -> float:
    """Approximate trend slope as percentage change over `window` bars."""
    if len(series) < window + 1:
        return np.nan
    try:
        a = float(series.iloc[-window - 1])
        b = float(series.iloc[-1])
        if a == 0:
            return np.nan
        return round((b / a - 1) * 100, 2)
    except Exception:
        return np.nan

def dist_from_ma(price: float, ma: float) -> float:
    if not ma or ma <= 0 or not np.isfinite(price) or not np.isfinite(ma):
        return np.nan
    return round((price / ma - 1) * 100, 2)

# =============================
# RALLY STAGE CLASSIFIER
# =============================

def classify_stage(m: dict) -> tuple:
    """
    Classify a stock's rally stage from a metrics dict produced in scan_market.
    Returns (stage_id, stage_label). Stages are evaluated 4 → 1 (extended first,
    then breakout, early markup, accumulation) so the most advanced setup wins.
    """
    price    = m.get("price")
    ma50     = m.get("ma50")
    ma200    = m.get("ma200")
    ma50_slope10 = m.get("ma50_slope10")
    ma200_slope20 = m.get("ma200_slope20")
    rsi      = m.get("rsi")
    adx      = m.get("adx")
    adx_prev = m.get("adx_prev")
    macd_hist = m.get("macd_hist")
    macd_above_signal = m.get("macd_above_signal")
    macd_cross_recent = m.get("macd_cross_recent")
    obv_slope = m.get("obv_slope")
    rs_nifty = m.get("rs_nifty")
    vol_ratio = m.get("vol_ratio")
    high20 = m.get("high20")
    bb_squeeze = m.get("bb_squeeze")
    dist_ma200 = m.get("dist_ma200")
    dist_ma50  = m.get("dist_ma50")
    is_breakout = m.get("is_breakout")

    def _f(x):
        try: return float(x)
        except Exception: return np.nan

    # Stage 4 — Extended / climax
    s4_signals = []
    if np.isfinite(_f(dist_ma50)) and _f(dist_ma50) > 25:
        s4_signals.append(True)
    if np.isfinite(_f(rsi)) and _f(rsi) > 75:
        s4_signals.append(True)
    if np.isfinite(_f(dist_ma200)) and _f(dist_ma200) > 60:
        s4_signals.append(True)
    if np.isfinite(_f(adx)) and _f(adx) > 50:
        s4_signals.append(True)
    if any(s4_signals):
        return 4, STAGE_LABELS[4]

    # Stage 3 — Breakout
    if is_breakout and np.isfinite(_f(macd_hist)) and _f(macd_hist) > 0 \
        and np.isfinite(_f(adx)) and _f(adx) > 25 \
        and np.isfinite(_f(vol_ratio)) and _f(vol_ratio) > 1.5 \
        and np.isfinite(_f(rsi)) and 55 <= _f(rsi) <= 72:
        return 3, STAGE_LABELS[3]

    # Stage 2 — Early Markup
    cond_s2 = (
        np.isfinite(_f(price)) and np.isfinite(_f(ma50)) and np.isfinite(_f(ma200))
        and _f(price) > _f(ma50) > _f(ma200)
        and np.isfinite(_f(ma50_slope10)) and _f(ma50_slope10) > 0
        and (macd_above_signal or macd_cross_recent)
        and np.isfinite(_f(adx)) and 18 <= _f(adx) <= 30
        and (not np.isfinite(_f(adx_prev)) or _f(adx) >= _f(adx_prev))
        and np.isfinite(_f(rsi)) and 50 <= _f(rsi) <= 65
        and np.isfinite(_f(high20)) and _f(price) < _f(high20) * BREAKOUT_TOLERANCE
        and np.isfinite(_f(rs_nifty)) and _f(rs_nifty) > 1.0
    )
    if cond_s2:
        return 2, STAGE_LABELS[2]

    # Stage 1 — Accumulation
    cond_s1 = (
        np.isfinite(_f(price)) and np.isfinite(_f(ma200))
        and abs(_f(price) / _f(ma200) - 1) <= 0.08
        and (not np.isfinite(_f(ma200_slope20)) or _f(ma200_slope20) >= -1.0)
        and bool(bb_squeeze)
        and np.isfinite(_f(adx)) and _f(adx) < 20
        and (not np.isfinite(_f(obv_slope)) or _f(obv_slope) >= 0)
        and np.isfinite(_f(rsi)) and 40 <= _f(rsi) <= 55
    )
    if cond_s1:
        return 1, STAGE_LABELS[1]

    return 0, STAGE_LABELS[0]

def estimate_upside(df: pd.DataFrame, price: float, high20: float) -> dict:
    high52 = df["High"].iloc[-252:].max() if len(df) >= 252 else df["High"].max()
    low52  = df["Low"].iloc[-252:].min()  if len(df) >= 252 else df["Low"].min()
    atr_target = price + 2 * (compute_atr_pct(df) / 100 * price)
    fib_target = low52 + (high52 - low52) * 1.618
    targets    = [t for t in [high52, atr_target, fib_target] if t > price * 1.02]
    target     = round(min(targets), 2) if targets else round(price * 1.12, 2)
    return {
        "target"        : target,
        "upside_pct"    : round((target - price) / price * 100, 1),
        "high52"        : round(high52, 2),
        "high52_gap_pct": round((high52 - price) / price * 100, 1),
    }

# =============================
# ETF SCANNER — dynamic registry, sanity-checked data, live progress
# =============================

def scan_etfs(etf_registry: pd.DataFrame, progress_cb=None) -> pd.DataFrame:
    """Fetch live price, RSI, trend, returns for every ETF. Reports progress via progress_cb."""
    if etf_registry.empty:
        return pd.DataFrame()

    tickers  = etf_registry["YF_SYMBOL"].tolist()
    name_map = dict(zip(etf_registry["YF_SYMBOL"], etf_registry["NAME"]))
    results  = []
    total    = len(tickers)

    for i in range(0, total, 50):
        chunk = tickers[i : i + 50]
        done  = min(i + 50, total)
        if progress_cb:
            try: progress_cb("etf", done, total, f"ETFs {done}/{total}")
            except Exception: pass
        try:
            data = _yf_download(
                chunk, period="1y", interval="1d",
                group_by="ticker", threads=True, progress=False, auto_adjust=True,
            )
        except Exception:
            continue

        for yf_sym in chunk:
            symbol = yf_sym.replace(".NS", "")
            try:
                df = data[yf_sym] if len(chunk) > 1 else data
                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Close"])
                if len(df) < 30:
                    continue

                price_series = df["Close"].dropna()
                if len(price_series) < 30:
                    continue

                price = float(price_series.iloc[-1])

                # Sanity: valid price range
                if price <= 0.5 or price > 500_000:
                    continue
                # Sanity: no >40% move in 5 days
                price_5d_ago = float(price_series.iloc[-6]) if len(price_series) >= 6 else price
                if price_5d_ago > 0 and abs(price / price_5d_ago - 1) > 0.4:
                    continue

                ma50 = float(price_series.rolling(50).mean().iloc[-1]) if len(price_series) >= 50 else np.nan
                rsi  = compute_rsi(price_series)

                # Sanity: realistic RSI range
                if np.isnan(rsi) or rsi < 10 or rsi > 90:
                    continue

                ret_1w = round((price / float(price_series.iloc[-6])  - 1) * 100, 2) if len(price_series) >= 6  else np.nan
                ret_1m = round((price / float(price_series.iloc[-22]) - 1) * 100, 2) if len(price_series) >= 22 else np.nan
                ret_3m = round((price / float(price_series.iloc[-65]) - 1) * 100, 2) if len(price_series) >= 65 else np.nan

                # Sanity: realistic returns for an ETF
                if any(abs(v) > 50 for v in [ret_1w, ret_1m, ret_3m] if not np.isnan(v)):
                    continue

                trend = "↑ Above MA50" if (not np.isnan(ma50) and price > ma50) else "↓ Below MA50"
                results.append({
                    "ETF"    : symbol,
                    "Name"   : name_map.get(yf_sym, symbol),
                    "Price ₹": round(price, 2),
                    "RSI"    : round(rsi, 1),
                    "Trend"  : trend,
                    "1W %"   : ret_1w,
                    "1M %"   : ret_1m,
                    "3M %"   : ret_3m,
                })
            except Exception:
                continue

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("1M %", ascending=False).reset_index(drop=True)


def build_etf_context(etf_df: pd.DataFrame) -> str:
    if etf_df.empty:
        return "ETF data unavailable."
    # Only ETFs with valid price > ₹1 and valid 1M return; top 15 by absolute 1M momentum
    valid = etf_df[
        (etf_df["Price ₹"] > 1) &
        (etf_df["1M %"].notna()) &
        (etf_df["RSI"].notna())
    ].copy()
    if valid.empty:
        valid = etf_df
    top = valid.reindex(valid["1M %"].abs().sort_values(ascending=False).index).head(15)
    lines = []
    for _, r in top.iterrows():
        rsi_flag = "[OB]" if r["RSI"] > RSI_OVERBOUGHT else ("[OS]" if r["RSI"] < 35 else "")
        lines.append(
            f"{r['ETF']} ({r['Name'][:30]}): ₹{r['Price ₹']} RSI{r['RSI']:.0f}{rsi_flag} "
            f"{r['Trend'].split()[0]} 1M{r['1M %']:+.1f}% 3M{r['3M %']:+.1f}%"
        )
    return "\n".join(lines)

# =============================
# MOMENTUM SCANNER
# =============================

@st.cache_data(ttl=3600, show_spinner=False)
def get_nifty_close(period: str = "1y") -> pd.Series:
    """Fetch ^NSEI close series. Cached 1h so the ad-hoc Search Stock tab
    doesn't refetch the index on every lookup."""
    try:
        hist = _yf_ticker("^NSEI").history(period=period, interval="1d", auto_adjust=True)
        if hist.empty:
            return pd.Series(dtype=float)
        return hist["Close"].dropna()
    except Exception:
        return pd.Series(dtype=float)

def scan_market(symbols: list, progress_cb=None, nifty_close: pd.Series | None = None) -> pd.DataFrame:
    results = []
    failed  = []
    total   = len(symbols)

    for i in range(0, total, CHUNK_SIZE):
        tickers = symbols[i : i + CHUNK_SIZE]
        current = min(i + CHUNK_SIZE, total)
        if progress_cb:
            try: progress_cb("scan", current, total, f"Equities {current}/{total}")
            except Exception: pass
        try:
            data = _yf_download(
                tickers, period="1y", interval="1d",
                group_by="ticker", threads=True, progress=False, auto_adjust=True,
            )
        except Exception as e:
            failed.extend([(t, str(e)) for t in tickers])
            continue

        for ticker in tickers:
            try:
                df = data[ticker] if len(tickers) > 1 else data
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.dropna(subset=["Close", "Volume"])
                if len(df) < MIN_BARS:
                    continue
                close = df["Close"]
                price        = float(close.iloc[-1])
                ma50         = compute_ma(close, MA_PERIOD)
                ma200        = compute_ma(close, MA_LONG_PERIOD)
                ma50_series  = close.rolling(MA_PERIOD).mean()
                ma200_series = close.rolling(MA_LONG_PERIOD).mean()
                ma50_slope10  = slope_pct(ma50_series.dropna(),  window=10)  if ma50_series.notna().sum()  >= 11 else np.nan
                ma200_slope20 = slope_pct(ma200_series.dropna(), window=20)  if ma200_series.notna().sum() >= 21 else np.nan

                high20    = float(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
                vol_avg   = float(df["Volume"].iloc[-(VOL_LOOKBACK + 1):-1].mean())
                vol_now   = float(df["Volume"].iloc[-1])
                vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
                rsi       = compute_rsi(close)
                atr_pct   = compute_atr_pct(df)
                upside    = estimate_upside(df, price, high20)

                macd_last, sig_last, hist_last, hist_series = compute_macd(close)
                macd_above_signal = bool(np.isfinite(macd_last) and np.isfinite(sig_last) and macd_last > sig_last)
                macd_cross_recent = False
                if len(hist_series.dropna()) >= 11:
                    tail = hist_series.dropna().iloc[-11:]
                    macd_cross_recent = bool(((tail.shift(1) < 0) & (tail > 0)).any())

                adx_series = compute_adx_series(df)
                if len(adx_series.dropna()) >= 6:
                    adx_now  = _safe_last(adx_series)
                    adx_prev = float(adx_series.dropna().iloc[-6])
                else:
                    adx_now, adx_prev = np.nan, np.nan
                _, plus_di, minus_di = compute_adx(df)

                roc20      = compute_roc(close, 20)
                obv_slope  = compute_obv_slope(close, df["Volume"])
                rs_nifty   = compute_rs_vs_nifty(close, nifty_close) if nifty_close is not None and not nifty_close.empty else np.nan
                bb_squeeze = is_bb_squeeze(close)
                d_ma50     = dist_from_ma(price, ma50)
                d_ma200    = dist_from_ma(price, ma200)

                is_breakout  = price >= high20 * BREAKOUT_TOLERANCE
                is_vol_surge = vol_ratio > VOL_SURGE_THRESH

                metrics = {
                    "price": price, "ma50": ma50, "ma200": ma200,
                    "ma50_slope10": ma50_slope10, "ma200_slope20": ma200_slope20,
                    "rsi": rsi, "adx": adx_now, "adx_prev": adx_prev,
                    "macd_hist": hist_last, "macd_above_signal": macd_above_signal,
                    "macd_cross_recent": macd_cross_recent,
                    "obv_slope": obv_slope, "rs_nifty": rs_nifty,
                    "vol_ratio": vol_ratio, "high20": high20,
                    "bb_squeeze": bb_squeeze,
                    "dist_ma200": d_ma200, "dist_ma50": d_ma50,
                    "is_breakout": is_breakout,
                }
                stage_id, stage_lbl = classify_stage(metrics)

                # Keep the row if EITHER the stage is meaningful (1-3) OR the
                # legacy breakout/volume condition fires above MA50. Stage 4
                # rows are still emitted (downweighted by score later).
                stage_pass  = stage_id in (1, 2, 3)
                legacy_pass = (np.isfinite(ma50) and price > ma50) and (is_breakout or is_vol_surge)
                if not (stage_pass or legacy_pass or stage_id == 4):
                    continue

                signal = "Breakout" if is_breakout else ("Building" if is_vol_surge else stage_lbl)

                results.append({
                    "Ticker"       : ticker.replace(".NS", ""),
                    "Price ₹"      : round(price, 2),
                    "MA50"         : round(ma50, 2) if np.isfinite(ma50) else np.nan,
                    "MA200"        : round(ma200, 2) if np.isfinite(ma200) else np.nan,
                    "RSI"          : rsi,
                    "Vol Ratio"    : round(vol_ratio, 2),
                    "Signal"       : signal,
                    "Stage"        : stage_lbl,
                    "StageId"      : stage_id,
                    "ADX"          : round(adx_now, 1) if np.isfinite(adx_now) else np.nan,
                    "+DI"          : round(plus_di, 1) if np.isfinite(plus_di) else np.nan,
                    "-DI"          : round(minus_di, 1) if np.isfinite(minus_di) else np.nan,
                    "MACD Hist"    : round(hist_last, 3) if np.isfinite(hist_last) else np.nan,
                    "ROC20 %"      : roc20,
                    "OBV Slope"    : obv_slope,
                    "RS vs Nifty"  : rs_nifty,
                    "Dist MA50 %"  : d_ma50,
                    "Dist MA200 %" : d_ma200,
                    "ATR %"        : atr_pct,
                    "Target ₹"     : upside["target"],
                    "Upside %"     : upside["upside_pct"],
                    "52W High ₹"   : upside["high52"],
                    "Gap to 52W %" : upside["high52_gap_pct"],
                })
            except Exception as e:
                failed.append((ticker, str(e)))

    if failed and progress_cb:
        try: progress_cb("scan", total, total, f"Done — {len(failed)} skipped (e.g. {[f[0] for f in failed[:3]]})")
        except Exception: pass
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df = compute_composite_score(df)
    return df.sort_values("Score", ascending=False).reset_index(drop=True)


# =============================
# SINGLE-STOCK TECHNICALS (used by the ad-hoc Search Stock tab)
# Mirrors scan_market's per-ticker work but for one symbol on demand,
# without any stage/breakout filter — every symbol gets a row.
# =============================

@st.cache_data(ttl=3600, show_spinner=False)
def compute_single_stock_technicals(symbol_yf: str, _nifty_hash: int = 0) -> dict | None:
    """One-stock equivalent of scan_market's per-row work. Returns a dict
    matching the scan_market row schema (including Score), or None if the
    data is too thin to be useful. `_nifty_hash` is just a cache key
    discriminator that changes with the cached Nifty series."""
    try:
        data = _yf_download(
            [symbol_yf], period="1y", interval="1d",
            group_by="ticker", threads=False, progress=False, auto_adjust=True,
        )
        if data is None or data.empty:
            return None
        # yfinance returns either a flat DataFrame (single ticker) or a
        # MultiIndex one; normalise to flat columns.
        if isinstance(data.columns, pd.MultiIndex):
            try:
                data = data[symbol_yf]
            except KeyError:
                data.columns = data.columns.get_level_values(0)
        df = data.dropna(subset=["Close", "Volume"])
        if len(df) < MIN_BARS:
            return None
    except Exception:
        return None

    nifty_close = get_nifty_close()

    close        = df["Close"]
    price        = float(close.iloc[-1])
    ma50         = compute_ma(close, MA_PERIOD)
    ma200        = compute_ma(close, MA_LONG_PERIOD)
    ma50_series  = close.rolling(MA_PERIOD).mean()
    ma200_series = close.rolling(MA_LONG_PERIOD).mean()
    ma50_slope10  = slope_pct(ma50_series.dropna(),  window=10)  if ma50_series.notna().sum()  >= 11 else np.nan
    ma200_slope20 = slope_pct(ma200_series.dropna(), window=20)  if ma200_series.notna().sum() >= 21 else np.nan

    high20    = float(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
    vol_avg   = float(df["Volume"].iloc[-(VOL_LOOKBACK + 1):-1].mean())
    vol_now   = float(df["Volume"].iloc[-1])
    vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
    rsi       = compute_rsi(close)
    atr_pct   = compute_atr_pct(df)
    upside    = estimate_upside(df, price, high20)

    macd_last, sig_last, hist_last, hist_series = compute_macd(close)
    macd_above_signal = bool(np.isfinite(macd_last) and np.isfinite(sig_last) and macd_last > sig_last)
    macd_cross_recent = False
    if len(hist_series.dropna()) >= 11:
        tail = hist_series.dropna().iloc[-11:]
        macd_cross_recent = bool(((tail.shift(1) < 0) & (tail > 0)).any())

    adx_series = compute_adx_series(df)
    if len(adx_series.dropna()) >= 6:
        adx_now  = _safe_last(adx_series)
        adx_prev = float(adx_series.dropna().iloc[-6])
    else:
        adx_now, adx_prev = np.nan, np.nan
    _, plus_di, minus_di = compute_adx(df)

    roc20      = compute_roc(close, 20)
    obv_slope  = compute_obv_slope(close, df["Volume"])
    rs_nifty   = compute_rs_vs_nifty(close, nifty_close) if nifty_close is not None and not nifty_close.empty else np.nan
    bb_squeeze = is_bb_squeeze(close)
    d_ma50     = dist_from_ma(price, ma50)
    d_ma200    = dist_from_ma(price, ma200)

    is_breakout  = price >= high20 * BREAKOUT_TOLERANCE
    is_vol_surge = vol_ratio > VOL_SURGE_THRESH

    classify_input = {
        "price": price, "ma50": ma50, "ma200": ma200,
        "ma50_slope10": ma50_slope10, "ma200_slope20": ma200_slope20,
        "rsi": rsi, "adx": adx_now, "adx_prev": adx_prev,
        "macd_hist": hist_last, "macd_above_signal": macd_above_signal,
        "macd_cross_recent": macd_cross_recent,
        "obv_slope": obv_slope, "rs_nifty": rs_nifty,
        "vol_ratio": vol_ratio, "high20": high20,
        "bb_squeeze": bb_squeeze,
        "dist_ma200": d_ma200, "dist_ma50": d_ma50,
        "is_breakout": is_breakout,
    }
    stage_id, stage_lbl = classify_stage(classify_input)
    signal = "Breakout" if is_breakout else ("Building" if is_vol_surge else stage_lbl)

    row = {
        "Ticker"       : symbol_yf.replace(".NS", ""),
        "Price ₹"      : round(price, 2),
        "MA50"         : round(ma50, 2) if np.isfinite(ma50) else np.nan,
        "MA200"        : round(ma200, 2) if np.isfinite(ma200) else np.nan,
        "RSI"          : rsi,
        "Vol Ratio"    : round(vol_ratio, 2),
        "Signal"       : signal,
        "Stage"        : stage_lbl,
        "StageId"      : stage_id,
        "ADX"          : round(adx_now, 1) if np.isfinite(adx_now) else np.nan,
        "+DI"          : round(plus_di, 1) if np.isfinite(plus_di) else np.nan,
        "-DI"          : round(minus_di, 1) if np.isfinite(minus_di) else np.nan,
        "MACD Hist"    : round(hist_last, 3) if np.isfinite(hist_last) else np.nan,
        "ROC20 %"      : roc20,
        "OBV Slope"    : obv_slope,
        "RS vs Nifty"  : rs_nifty,
        "Dist MA50 %"  : d_ma50,
        "Dist MA200 %" : d_ma200,
        "ATR %"        : atr_pct,
        "Target ₹"     : upside["target"],
        "Upside %"     : upside["upside_pct"],
        "52W High ₹"   : upside["high52"],
        "Gap to 52W %" : upside["high52_gap_pct"],
    }
    # Single-row score via the same composite engine (with std fallback).
    scored = compute_composite_score(pd.DataFrame([row]))
    row["Score"] = float(scored.iloc[0]["Score"]) if not scored.empty else np.nan
    return row


# =============================
# COMPOSITE SCORING
# =============================

def _rsi_health(r):
    if not np.isfinite(r):
        return 0.4
    if 50 <= r <= 65:
        return 1.0
    if 40 <= r < 50 or 65 < r <= 72:
        return 0.8
    return 0.4

def compute_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 0–100 Score column blending stage, momentum, trend & RS."""
    if df.empty:
        return df
    d = df.copy()
    stage_w  = d["StageId"].map(STAGE_WEIGHTS).fillna(0.2)
    vol_w    = (d["Vol Ratio"].fillna(0) / 3.0).clip(0, 1)
    rsi_w    = d["RSI"].apply(_rsi_health)
    hist     = d["MACD Hist"].fillna(0.0)
    # std() returns NaN for a 1-row frame and 0 if all rows are equal — both
    # would break the score. Fall back to 1.0 so a single-row scoring
    # (Search Stock tab) gets a meaningful value.
    raw_std  = hist.std()
    hist_std = raw_std if (pd.notna(raw_std) and raw_std > 0) else 1.0
    macd_w   = (hist / hist_std).clip(lower=0, upper=1)
    adx_w    = ((d["ADX"].fillna(0) - 15).clip(lower=0) / 25).clip(upper=1)
    rs_w     = ((d["RS vs Nifty"].fillna(1.0) - 1.0) * 5).clip(0, 1)
    up_w     = (d["Upside %"].fillna(0) / 30).clip(0, 1)

    score = 100 * (
        0.30 * stage_w +
        0.15 * vol_w   +
        0.10 * rsi_w   +
        0.15 * macd_w  +
        0.15 * adx_w   +
        0.10 * rs_w    +
        0.05 * up_w
    )
    d["Score"] = score.round(1)
    return d

def build_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """Top N rows above score threshold, with a floor."""
    if df.empty or "Score" not in df.columns:
        return df.head(0)
    above = df[df["Score"] >= SHORTLIST_MIN_SCORE].sort_values("Score", ascending=False)
    if len(above) >= SHORTLIST_FLOOR:
        return above.head(SHORTLIST_TARGET).reset_index(drop=True)
    # Fall back: take top SHORTLIST_FLOOR overall
    return df.sort_values("Score", ascending=False).head(SHORTLIST_FLOOR).reset_index(drop=True)

# =============================
# FUNDAMENTALS LAYER (free, yfinance-only — FMP-equivalent)
# =============================

def _safe(fn, default=None):
    try:
        v = fn()
        if v is None:
            return default
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return default
        return v
    except Exception:
        return default

def _first_row(df_like, candidates):
    """Return Series for first matching row label found in candidates."""
    if df_like is None or len(df_like) == 0:
        return None
    if not hasattr(df_like, "index"):
        return None
    idx = [str(x) for x in df_like.index]
    for cand in candidates:
        for i, label in enumerate(idx):
            if cand.lower() in label.lower():
                try:
                    return df_like.iloc[i]
                except Exception:
                    continue
    return None

def _cagr(series_values, years):
    try:
        vals = [float(v) for v in series_values if v is not None and np.isfinite(float(v))]
        if len(vals) < 2 or years < 1:
            return None
        oldest, latest = vals[-1], vals[0]   # yfinance returns newest first
        if oldest <= 0 or latest <= 0:
            return None
        return (latest / oldest) ** (1 / years) - 1
    except Exception:
        return None

def compute_dcf(info: dict, cashflow: pd.DataFrame, balance: pd.DataFrame) -> dict:
    """
    Simple 5-yr DCF: project FCF using historical CAGR (clipped 4-18%),
    fall back to info.earningsGrowth, else 10%. Discount 12%, terminal 4%.
    Returns dict with intrinsic_per_share, upside_pct, assumptions, confidence.
    """
    out = {"intrinsic_per_share": None, "upside_pct": None,
           "assumptions": {}, "confidence": "low"}

    fcf_row = _first_row(cashflow, ["Free Cash Flow"])
    if fcf_row is None:
        ocf = _first_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = _first_row(cashflow, ["Capital Expenditure"])
        if ocf is not None and capex is not None:
            try:
                fcf_row = ocf.add(capex, fill_value=0)  # capex is negative in yfinance
            except Exception:
                fcf_row = None

    if fcf_row is None or fcf_row.dropna().empty:
        return out

    fcf_values = [float(v) for v in fcf_row.dropna().tolist() if np.isfinite(float(v))]
    if not fcf_values:
        return out

    base_fcf = fcf_values[0]
    if base_fcf <= 0:
        return out

    growth = _cagr(fcf_values, min(3, len(fcf_values) - 1))
    if growth is None:
        growth = _safe(lambda: float(info.get("earningsGrowth")))
    if growth is None:
        growth = 0.10
    growth = max(0.04, min(0.18, float(growth)))

    r, g = 0.12, 0.04
    discounted = 0.0
    proj_fcf = base_fcf
    for t in range(1, 6):
        proj_fcf = proj_fcf * (1 + growth)
        discounted += proj_fcf / ((1 + r) ** t)
    terminal = proj_fcf * (1 + g) / (r - g)
    ev = discounted + terminal / ((1 + r) ** 5)

    total_debt = _safe(lambda: float(info.get("totalDebt")), 0.0) or 0.0
    cash       = _safe(lambda: float(info.get("totalCash")), 0.0) or 0.0
    shares     = _safe(lambda: float(info.get("sharesOutstanding")))
    price      = _safe(lambda: float(info.get("currentPrice") or info.get("regularMarketPrice")))

    if not shares or shares <= 0:
        return out

    equity_value = ev - total_debt + cash
    intrinsic = equity_value / shares
    upside_pct = (intrinsic - price) / price * 100 if price and price > 0 else None

    confidence = "high" if len(fcf_values) >= 4 else ("medium" if len(fcf_values) >= 3 else "low")

    out.update({
        "intrinsic_per_share": round(intrinsic, 2),
        "upside_pct": round(upside_pct, 1) if upside_pct is not None else None,
        "assumptions": {
            "base_fcf": base_fcf, "growth": round(growth, 3),
            "discount_rate": r, "terminal_growth": g,
            "fcf_history_years": len(fcf_values),
        },
        "confidence": confidence,
    })
    return out

@st.cache_data(ttl=FUND_CACHE_TTL_SEC, show_spinner=False)
def fetch_fundamentals(symbol: str) -> dict:
    """Pull FMP-equivalent fundamentals from yfinance for a single ticker."""
    out = {"symbol": symbol, "fetched_at": fmt_ist("%Y-%m-%d %H:%M IST")}
    try:
        t = _yf_ticker(symbol)
        info = _safe(lambda: t.info, default={}) or {}
        fin  = _safe(lambda: t.financials,  default=pd.DataFrame())
        bal  = _safe(lambda: t.balance_sheet, default=pd.DataFrame())
        cf   = _safe(lambda: t.cashflow,    default=pd.DataFrame())
        eh   = _safe(lambda: t.earnings_history, default=pd.DataFrame())
        rec  = _safe(lambda: t.recommendations_summary, default=pd.DataFrame())
        news = _safe(lambda: t.news, default=[]) or []
    except Exception as e:
        out["error"] = str(e)
        return out

    # ── Identity ───────────────────────────────────────────────
    out["name"]     = _safe(lambda: info.get("longName") or info.get("shortName"))
    out["sector"]   = _safe(lambda: info.get("sector"))
    out["industry"] = _safe(lambda: info.get("industry"))
    out["country"]  = _safe(lambda: info.get("country"))
    out["price"]    = _safe(lambda: float(info.get("currentPrice") or info.get("regularMarketPrice")))
    out["market_cap"] = _safe(lambda: float(info.get("marketCap")))

    # ── Valuation ─────────────────────────────────────────────
    out["pe_trailing"]    = _safe(lambda: float(info.get("trailingPE")))
    out["pe_forward"]     = _safe(lambda: float(info.get("forwardPE")))
    out["pb"]             = _safe(lambda: float(info.get("priceToBook")))
    out["ev_ebitda"]      = _safe(lambda: float(info.get("enterpriseToEbitda")))
    out["peg"]            = _safe(lambda: float(info.get("pegRatio")))
    out["price_to_sales"] = _safe(lambda: float(info.get("priceToSalesTrailing12Months")))
    out["dividend_yield"] = _safe(lambda: float(info.get("dividendYield")))

    # ── Quality / Profitability ───────────────────────────────
    out["roe"]            = _safe(lambda: float(info.get("returnOnEquity")))
    out["roa"]            = _safe(lambda: float(info.get("returnOnAssets")))
    out["gross_margin"]   = _safe(lambda: float(info.get("grossMargins")))
    out["op_margin"]      = _safe(lambda: float(info.get("operatingMargins")))
    out["net_margin"]     = _safe(lambda: float(info.get("profitMargins")))
    out["debt_to_equity"] = _safe(lambda: float(info.get("debtToEquity")) / 100.0
                                  if info.get("debtToEquity") is not None else None)
    out["current_ratio"]  = _safe(lambda: float(info.get("currentRatio")))
    out["quick_ratio"]    = _safe(lambda: float(info.get("quickRatio")))

    # Derived: ROCE, ROIC, Interest Coverage
    ebit_row   = _first_row(fin, ["EBIT", "Operating Income"])
    intexp_row = _first_row(fin, ["Interest Expense"])
    ta_row     = _first_row(bal, ["Total Assets"])
    cl_row     = _first_row(bal, ["Current Liabilities", "Total Current Liabilities"])
    eq_row     = _first_row(bal, ["Total Stockholder Equity", "Stockholders Equity", "Common Stock Equity"])
    ltd_row    = _first_row(bal, ["Long Term Debt"])
    cash_row   = _first_row(bal, ["Cash And Cash Equivalents", "Cash"])

    def _latest(row):
        try:
            v = float(row.dropna().iloc[0])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    ebit_v = _latest(ebit_row) if ebit_row is not None else None
    ta_v   = _latest(ta_row)   if ta_row   is not None else None
    cl_v   = _latest(cl_row)   if cl_row   is not None else None
    eq_v   = _latest(eq_row)   if eq_row   is not None else None
    ltd_v  = _latest(ltd_row)  if ltd_row  is not None else None
    cash_v = _latest(cash_row) if cash_row is not None else None
    intexp_v = _latest(intexp_row) if intexp_row is not None else None

    if ebit_v and ta_v and cl_v and (ta_v - cl_v) != 0:
        out["roce"] = round(ebit_v / (ta_v - cl_v), 4)
    else:
        out["roce"] = None

    if ebit_v and eq_v:
        invested = (eq_v or 0) + (ltd_v or 0) - (cash_v or 0)
        if invested > 0:
            out["roic"] = round((ebit_v * (1 - 0.25)) / invested, 4)
        else:
            out["roic"] = None
    else:
        out["roic"] = None

    if ebit_v and intexp_v and intexp_v != 0:
        out["interest_coverage"] = round(ebit_v / abs(intexp_v), 2)
    else:
        out["interest_coverage"] = None

    # ── Growth ───────────────────────────────────────────────
    rev_row = _first_row(fin, ["Total Revenue", "Revenue"])
    ni_row  = _first_row(fin, ["Net Income"])
    shares  = _safe(lambda: float(info.get("sharesOutstanding")))

    if rev_row is not None and not rev_row.dropna().empty:
        rev_vals = [float(v) for v in rev_row.dropna().tolist() if np.isfinite(float(v))]
        if len(rev_vals) >= 4:
            out["rev_cagr_3y"] = _cagr(rev_vals[:4], 3)
        if len(rev_vals) >= 5:
            out["rev_cagr_5y"] = _cagr(rev_vals[:5], 4)
        out["rev_latest"] = rev_vals[0] if rev_vals else None
    if ni_row is not None and not ni_row.dropna().empty and shares:
        ni_vals = [float(v) for v in ni_row.dropna().tolist() if np.isfinite(float(v))]
        eps_vals = [v / shares for v in ni_vals if shares]
        if len(eps_vals) >= 4:
            out["eps_cagr_3y"] = _cagr(eps_vals[:4], 3)

    out["earnings_growth"] = _safe(lambda: float(info.get("earningsGrowth")))
    out["revenue_growth"]  = _safe(lambda: float(info.get("revenueGrowth")))
    out["fwd_eps_growth"]  = _safe(lambda: (float(info.get("forwardEps")) / float(info.get("trailingEps")) - 1)
                                   if info.get("forwardEps") and info.get("trailingEps") else None)

    try:
        if isinstance(eh, pd.DataFrame) and not eh.empty:
            col = next((c for c in eh.columns if "surprisePercent" in c.lower() or "surprise" in c.lower()), None)
            if col:
                surprises = eh[col].dropna().head(4).astype(float).tolist()
                if surprises:
                    out["earnings_surprise_avg"] = round(float(np.mean(surprises)), 2)
    except Exception:
        pass

    # ── Ownership & Sentiment ───────────────────────────────
    out["insider_held"]      = _safe(lambda: float(info.get("heldPercentInsiders")))
    out["institutional_held"] = _safe(lambda: float(info.get("heldPercentInstitutions")))
    out["recommendation_key"] = _safe(lambda: info.get("recommendationKey"))
    out["recommendation_mean"] = _safe(lambda: float(info.get("recommendationMean")))
    out["analyst_count"]     = _safe(lambda: int(info.get("numberOfAnalystOpinions")))
    out["target_mean"]       = _safe(lambda: float(info.get("targetMeanPrice")))
    out["target_high"]       = _safe(lambda: float(info.get("targetHighPrice")))
    out["target_low"]        = _safe(lambda: float(info.get("targetLowPrice")))

    headlines = []
    for n in (news or [])[:5]:
        try:
            title = n.get("title") or n.get("content", {}).get("title")
            if title:
                headlines.append(str(title))
        except Exception:
            continue
    out["news_headlines"] = headlines

    # ── DCF ───────────────────────────────────────────────
    out["dcf"] = compute_dcf(info, cf, bal)

    return out

def fetch_fundamentals_bulk(symbols: list, progress_cb=None) -> dict:
    """Parallel fetch for a list of YF symbols (e.g. 'TCS.NS'). Returns dict by symbol."""
    out = {}
    total = len(symbols)
    if total == 0:
        return out
    done = 0
    with ThreadPoolExecutor(max_workers=FUND_WORKERS) as pool:
        futures = {pool.submit(fetch_fundamentals, s): s for s in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                out[sym] = fut.result()
            except Exception as e:
                out[sym] = {"symbol": sym, "error": str(e)}
            done += 1
            if progress_cb:
                try: progress_cb("funds", done, total, f"Fundamentals {done}/{total}")
                except Exception: pass
    return out

# =============================
# RECOMMENDER
# =============================

def _pct(v, decimals=1):
    if v is None or not np.isfinite(v):
        return "N/A"
    return f"{v * 100:.{decimals}f}%"

def _num(v, decimals=2, prefix="", suffix=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "N/A"
    return f"{prefix}{v:,.{decimals}f}{suffix}"

def _coalesce(v, default):
    """Return default if v is None or NaN; else v."""
    try:
        if v is None:
            return default
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return default
        return v
    except Exception:
        return default

def recommend_action(metrics: dict, tech: dict) -> dict:
    """
    metrics = fundamentals dict from fetch_fundamentals.
    tech    = single-stock row from the momentum scan (Score, Stage, ATR %, RSI, ADX, RS, Upside %, etc.).
    Returns {action, conviction, bull_case, bear_case, entry_zone, stop_loss, targets}.
    """
    price = _coalesce(tech.get("Price ₹"), _coalesce(metrics.get("price"), 0))
    atrp  = _coalesce(tech.get("ATR %"), 3.0)
    try:
        stage_id = int(_coalesce(tech.get("StageId"), 0))
    except Exception:
        stage_id = 0

    # Component scores 0–100
    tech_score = float(_coalesce(tech.get("Score"), 0))

    qty_inputs = [
        (metrics.get("roe"),           lambda v: 1.0 if v and v > 0.18 else (0.6 if v and v > 0.12 else 0.2)),
        (metrics.get("op_margin"),     lambda v: 1.0 if v and v > 0.15 else (0.6 if v and v > 0.08 else 0.2)),
        (metrics.get("debt_to_equity"),lambda v: 0.2 if v and v > 1.5 else (0.6 if v and v > 0.7 else 1.0)),
        (metrics.get("interest_coverage"), lambda v: 1.0 if v and v > 6 else (0.5 if v and v > 3 else 0.1)),
    ]
    qparts = [f(v) for v, f in qty_inputs if v is not None]
    quality_score = 100 * (sum(qparts) / len(qparts)) if qparts else 50.0

    val_inputs = []
    pe = metrics.get("pe_forward") or metrics.get("pe_trailing")
    if pe and pe > 0:
        val_inputs.append(1.0 if pe < 20 else (0.7 if pe < 35 else (0.4 if pe < 60 else 0.1)))
    peg = metrics.get("peg")
    if peg and peg > 0:
        val_inputs.append(1.0 if peg < 1 else (0.6 if peg < 2 else 0.2))
    dcf_up = (metrics.get("dcf") or {}).get("upside_pct")
    if dcf_up is not None:
        val_inputs.append(1.0 if dcf_up > 20 else (0.6 if dcf_up > 0 else 0.2))
    value_score = 100 * (sum(val_inputs) / len(val_inputs)) if val_inputs else 50.0

    g_inputs = []
    rg = metrics.get("rev_cagr_3y")
    if rg is not None:
        g_inputs.append(1.0 if rg > 0.15 else (0.6 if rg > 0.07 else 0.2))
    eg = metrics.get("eps_cagr_3y")
    if eg is not None:
        g_inputs.append(1.0 if eg > 0.15 else (0.6 if eg > 0.07 else 0.2))
    fg = metrics.get("fwd_eps_growth")
    if fg is not None:
        g_inputs.append(1.0 if fg > 0.10 else (0.6 if fg > 0 else 0.2))
    growth_score = 100 * (sum(g_inputs) / len(g_inputs)) if g_inputs else 50.0

    conviction = 0.4 * tech_score + 0.3 * quality_score + 0.2 * value_score + 0.1 * growth_score
    conviction = round(max(0, min(100, conviction)), 1)

    def _gt(v, thr):
        v = _coalesce(v, None)
        return v is not None and v > thr
    def _lt(v, thr):
        v = _coalesce(v, None)
        return v is not None and v < thr

    # Bull / bear flags
    bull = []
    if _gt(metrics.get("roe"), 0.18):                    bull.append(f"ROE {metrics['roe']*100:.1f}% > 18%")
    if _gt(metrics.get("op_margin"), 0.15):              bull.append(f"Operating margin {metrics['op_margin']*100:.1f}% > 15%")
    if _gt(metrics.get("rev_cagr_3y"), 0.15):            bull.append(f"Revenue 3Y CAGR {metrics['rev_cagr_3y']*100:.1f}%")
    if _gt(dcf_up, 20):                                  bull.append(f"DCF upside {dcf_up:.1f}%")
    if stage_id in (1, 2):                               bull.append(f"Stage {stage_id} setup ({STAGE_LABELS[stage_id]})")
    if _gt(tech.get("RS vs Nifty"), 1.1):                bull.append(f"RS vs Nifty {tech['RS vs Nifty']:.2f}")
    if _gt(metrics.get("institutional_held"), 0.40):     bull.append(f"Institutional hold {metrics['institutional_held']*100:.0f}%")
    if _gt(tech.get("ADX"), 25):                         bull.append(f"ADX {tech['ADX']:.0f} (trending)")
    if _gt(metrics.get("earnings_surprise_avg"), 5):     bull.append(f"Avg earnings surprise +{metrics['earnings_surprise_avg']:.1f}%")

    bear = []
    if _gt(metrics.get("debt_to_equity"), 1.5):          bear.append(f"D/E {metrics['debt_to_equity']:.2f} > 1.5")
    if _lt(metrics.get("interest_coverage"), 3):         bear.append(f"Interest cover {metrics['interest_coverage']:.1f}× weak")
    if _gt(pe, 50) and _gt(peg, 2):                      bear.append(f"Expensive: P/E {pe:.0f}, PEG {peg:.1f}")
    if stage_id == 4:                                    bear.append("Stage 4 — extended / climax risk")
    if _lt(metrics.get("rev_cagr_3y"), 0):               bear.append(f"Revenue declining ({metrics['rev_cagr_3y']*100:.1f}% 3Y CAGR)")
    if _lt(metrics.get("op_margin"), 0.05):              bear.append("Operating margin < 5%")
    if _lt(dcf_up, -20):                                 bear.append(f"DCF says overvalued ({dcf_up:.0f}%)")

    bull_case = bull[:3] if bull else ["No standout positives"]
    bear_case = bear[:3] if bear else ["No critical red flags"]

    critical_bear = any(b in bear for b in [
        f"D/E {metrics.get('debt_to_equity', 0):.2f} > 1.5",
        "Stage 4 — extended / climax risk",
    ])

    if stage_id == 4:
        action = "AVOID"
    elif conviction >= 80 and not critical_bear:
        action = "STRONG_BUY"
    elif conviction >= 65:
        action = "BUY"
    elif conviction >= 50:
        action = "ACCUMULATE"
    elif conviction >= 35:
        action = "HOLD"
    else:
        action = "AVOID"

    # Entry / stop / targets
    atrp_use = atrp if (isinstance(atrp, (int, float)) and np.isfinite(atrp) and atrp > 0) else 3.0
    price_v = price if (isinstance(price, (int, float)) and np.isfinite(price) and price > 0) else None
    entry_low  = round(price_v * (1 - 0.5 * atrp_use / 100), 2) if price_v else None
    entry_high = round(price_v, 2) if price_v else None
    stop       = round(price_v * (1 - 2 * atrp_use / 100), 2) if price_v else None

    targets = {}
    if tech.get("Target ₹"):
        targets["fib"] = float(tech["Target ₹"])
    if metrics.get("target_mean"):
        targets["analyst_mean"] = float(metrics["target_mean"])
    dcf_intrinsic = (metrics.get("dcf") or {}).get("intrinsic_per_share")
    if dcf_intrinsic:
        targets["dcf"] = float(dcf_intrinsic)

    return {
        "action": action,
        "conviction": conviction,
        "scores": {
            "technical": round(tech_score, 1),
            "quality":   round(quality_score, 1),
            "value":     round(value_score, 1),
            "growth":    round(growth_score, 1),
        },
        "bull_case": bull_case,
        "bear_case": bear_case,
        "entry_zone": (entry_low, entry_high),
        "stop_loss": stop,
        "targets": targets,
    }

def analyze_stock_fundamentals(symbol: str, technicals_row: dict) -> dict:
    """Fetch fundamentals (cached) and merge with technicals into a recommendation."""
    metrics = fetch_fundamentals(symbol)
    rec     = recommend_action(metrics, technicals_row)
    return {"symbol": symbol, "metrics": metrics, "technicals": technicals_row, "recommendation": rec}

# =============================
# EXCEL EXPORT
# =============================

def _flatten_metrics(sym_no_suffix: str, m: dict, t: dict, rec: dict) -> dict:
    """One flat row for the Summary sheet."""
    dcf = m.get("dcf") or {}
    return {
        "Ticker": sym_no_suffix,
        "Name": m.get("name"),
        "Sector": m.get("sector"),
        "Industry": m.get("industry"),
        "Price ₹": t.get("Price ₹"),
        "Stage": t.get("Stage"),
        "Score": t.get("Score"),
        "Action": rec.get("action"),
        "Conviction": rec.get("conviction"),
        "Upside % (tech)": t.get("Upside %"),
        "DCF Intrinsic ₹": dcf.get("intrinsic_per_share"),
        "DCF Upside %": dcf.get("upside_pct"),
        "P/E (fwd)": m.get("pe_forward"),
        "P/B": m.get("pb"),
        "EV/EBITDA": m.get("ev_ebitda"),
        "ROE %": (m.get("roe") or 0) * 100 if m.get("roe") is not None else None,
        "ROCE %": (m.get("roce") or 0) * 100 if m.get("roce") is not None else None,
        "Op Margin %": (m.get("op_margin") or 0) * 100 if m.get("op_margin") is not None else None,
        "D/E": m.get("debt_to_equity"),
        "Rev CAGR 3Y %": (m.get("rev_cagr_3y") or 0) * 100 if m.get("rev_cagr_3y") is not None else None,
        "Inst Hold %": (m.get("institutional_held") or 0) * 100 if m.get("institutional_held") is not None else None,
        "Analyst Rec": m.get("recommendation_key"),
    }

def build_excel_workbook(scan_df: pd.DataFrame, shortlist_df: pd.DataFrame, fund_map: dict) -> bytes:
    """
    Return XLSX bytes. Sheets:
      Full Scan, Summary, Valuation, Quality, Growth, Ownership, Recommendation,
      and one vertical sheet per shortlisted stock (cap 25).
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if scan_df is not None and not scan_df.empty:
            scan_df.to_excel(writer, sheet_name="Full Scan", index=False)

        rows_summary, rows_val, rows_q, rows_g, rows_o, rows_rec = [], [], [], [], [], []

        for _, t_row in shortlist_df.iterrows():
            ticker_no = t_row["Ticker"]
            yf_sym = f"{ticker_no}.NS"
            payload = fund_map.get(yf_sym) or {}
            m = payload if "metrics" not in payload else payload.get("metrics", {})
            if "metrics" in payload:
                m = payload["metrics"]
                rec = payload["recommendation"]
            else:
                rec = recommend_action(m, t_row.to_dict())

            rows_summary.append(_flatten_metrics(ticker_no, m, t_row.to_dict(), rec))
            dcf = m.get("dcf") or {}
            rows_val.append({
                "Ticker": ticker_no,
                "P/E trailing": m.get("pe_trailing"), "P/E forward": m.get("pe_forward"),
                "P/B": m.get("pb"), "EV/EBITDA": m.get("ev_ebitda"), "PEG": m.get("peg"),
                "P/S": m.get("price_to_sales"),
                "DCF Intrinsic ₹": dcf.get("intrinsic_per_share"),
                "DCF Upside %": dcf.get("upside_pct"),
                "DCF Confidence": dcf.get("confidence"),
                "Dividend Yield %": (m.get("dividend_yield") or 0) * 100 if m.get("dividend_yield") is not None else None,
            })
            rows_q.append({
                "Ticker": ticker_no,
                "ROE %": (m.get("roe") or 0) * 100 if m.get("roe") is not None else None,
                "ROA %": (m.get("roa") or 0) * 100 if m.get("roa") is not None else None,
                "ROCE %": (m.get("roce") or 0) * 100 if m.get("roce") is not None else None,
                "ROIC %": (m.get("roic") or 0) * 100 if m.get("roic") is not None else None,
                "Gross Margin %": (m.get("gross_margin") or 0) * 100 if m.get("gross_margin") is not None else None,
                "Op Margin %": (m.get("op_margin") or 0) * 100 if m.get("op_margin") is not None else None,
                "Net Margin %": (m.get("net_margin") or 0) * 100 if m.get("net_margin") is not None else None,
                "D/E": m.get("debt_to_equity"),
                "Interest Coverage": m.get("interest_coverage"),
                "Current Ratio": m.get("current_ratio"),
                "Quick Ratio": m.get("quick_ratio"),
            })
            rows_g.append({
                "Ticker": ticker_no,
                "Revenue CAGR 3Y %": (m.get("rev_cagr_3y") or 0) * 100 if m.get("rev_cagr_3y") is not None else None,
                "Revenue CAGR 5Y %": (m.get("rev_cagr_5y") or 0) * 100 if m.get("rev_cagr_5y") is not None else None,
                "EPS CAGR 3Y %": (m.get("eps_cagr_3y") or 0) * 100 if m.get("eps_cagr_3y") is not None else None,
                "Earnings Growth %": (m.get("earnings_growth") or 0) * 100 if m.get("earnings_growth") is not None else None,
                "Revenue Growth %": (m.get("revenue_growth") or 0) * 100 if m.get("revenue_growth") is not None else None,
                "Fwd EPS Growth %": (m.get("fwd_eps_growth") or 0) * 100 if m.get("fwd_eps_growth") is not None else None,
                "Avg Earnings Surprise %": m.get("earnings_surprise_avg"),
            })
            rows_o.append({
                "Ticker": ticker_no,
                "Insider Held %": (m.get("insider_held") or 0) * 100 if m.get("insider_held") is not None else None,
                "Institutional Held %": (m.get("institutional_held") or 0) * 100 if m.get("institutional_held") is not None else None,
                "Analyst Rec": m.get("recommendation_key"),
                "Rec Mean (1=Strong Buy)": m.get("recommendation_mean"),
                "Analyst Count": m.get("analyst_count"),
                "Target Mean ₹": m.get("target_mean"),
                "Target High ₹": m.get("target_high"),
                "Target Low ₹":  m.get("target_low"),
            })
            rows_rec.append({
                "Ticker": ticker_no,
                "Action": rec.get("action"),
                "Conviction": rec.get("conviction"),
                "Tech Score": rec.get("scores", {}).get("technical"),
                "Quality Score": rec.get("scores", {}).get("quality"),
                "Value Score": rec.get("scores", {}).get("value"),
                "Growth Score": rec.get("scores", {}).get("growth"),
                "Entry Low": rec.get("entry_zone", (None, None))[0],
                "Entry High": rec.get("entry_zone", (None, None))[1],
                "Stop Loss": rec.get("stop_loss"),
                "Target Fib": rec.get("targets", {}).get("fib"),
                "Target Analyst": rec.get("targets", {}).get("analyst_mean"),
                "Target DCF": rec.get("targets", {}).get("dcf"),
                "Bull Case": " | ".join(rec.get("bull_case", [])),
                "Bear Case": " | ".join(rec.get("bear_case", [])),
            })

        pd.DataFrame(rows_summary).to_excel(writer, sheet_name="Summary",        index=False)
        pd.DataFrame(rows_val).to_excel(writer,     sheet_name="Valuation",      index=False)
        pd.DataFrame(rows_q).to_excel(writer,       sheet_name="Quality",        index=False)
        pd.DataFrame(rows_g).to_excel(writer,       sheet_name="Growth",         index=False)
        pd.DataFrame(rows_o).to_excel(writer,       sheet_name="Ownership",      index=False)
        pd.DataFrame(rows_rec).to_excel(writer,     sheet_name="Recommendation", index=False)

        # Per-stock vertical sheets (cap 25)
        for _, t_row in shortlist_df.head(25).iterrows():
            ticker_no = t_row["Ticker"]
            yf_sym = f"{ticker_no}.NS"
            payload = fund_map.get(yf_sym) or {}
            m   = payload.get("metrics", payload) if isinstance(payload, dict) else {}
            rec = payload.get("recommendation") if isinstance(payload, dict) else None
            if rec is None:
                rec = recommend_action(m, t_row.to_dict())
            sheet_rows = list(_flatten_metrics(ticker_no, m, t_row.to_dict(), rec).items())
            sheet_rows.append(("Bull Case", " | ".join(rec.get("bull_case", []))))
            sheet_rows.append(("Bear Case", " | ".join(rec.get("bear_case", []))))
            df_sheet = pd.DataFrame(sheet_rows, columns=["Metric", "Value"])
            # Excel sheet names ≤31 chars, no special chars
            sn = re.sub(r"[^A-Za-z0-9_]", "_", ticker_no)[:28]
            df_sheet.to_excel(writer, sheet_name=sn, index=False)

    return buf.getvalue()

# =============================
# LIVE MARKET SNAPSHOT — fetch current Nifty, Sensex, INR from yfinance
# =============================

def get_live_market_snapshot() -> str:
    """Fetch real-time Nifty50, Bank Nifty, USD/INR, Brent crude levels."""
    symbols = {
        "^NSEI"  : "Nifty 50",
        "^NSEBANK": "Bank Nifty",
        "USDINR=X": "USD/INR",
        "BZ=F"   : "Brent Crude (USD/bbl)",
    }
    lines = []
    for sym, label in symbols.items():
        try:
            t    = _yf_ticker(sym)
            hist = t.history(period="2d", interval="1d")
            if hist.empty:
                continue
            price  = float(hist["Close"].iloc[-1])
            prev   = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            chg    = round((price / prev - 1) * 100, 2)
            lines.append(f"- {label}: {price:,.2f} ({chg:+.2f}% vs prev close)")
        except Exception:
            continue
    return "\n".join(lines) if lines else "Live data unavailable."


# =============================
# NEWS FETCH
# =============================

def get_raw_news() -> str:
    today = now_ist().strftime("%d %B %Y")
    query = (
        f"India stock market Nifty Sensex {today} RBI interest rate FII DII flows "
        f"Brent crude oil INR USD rupee NSE BSE "
        f"Nifty Bank IT pharma auto metals earnings sector"
    )
    lines = []
    try:
        result = tavily.search(query=query, max_results=8)
        for i, r in enumerate(result.get("results", []), 1):
            title   = r.get("title", "").strip()
            content = r.get("content", "").strip()
            if not title:
                continue
            lines.append(f"{i}. {title}\n   {content[:300]}")
    except Exception as e:
        lines.append(f"News fetch error: {e}")
    return "\n\n".join(lines) if lines else "No news retrieved."

# =============================
# AI MARKET INTELLIGENCE SUMMARY
# =============================

def generate_intel_summary(raw_news: str, live_snapshot: str) -> str:
    system = (
        "You are a senior Indian market analyst. Use the LIVE MARKET DATA as ground truth "
        "for current index levels and prices — never contradict it. Supplement with news context. "
        "Be specific with numbers. Do not reproduce headlines."
    )
    prompt = f"""Today: {now_ist().strftime('%d %B %Y, %H:%M IST')}

LIVE MARKET DATA (current prices as of now — use these exact numbers):
{live_snapshot}

NEWS CONTEXT:
{raw_news}

Write a Market Intelligence Brief. Use ## headers, **bold** for key numbers, bullet points.

## 🌏 Global Macro & Crude Oil
Current Brent crude level (from live data), US Fed stance, global risk sentiment.

## 🇮🇳 India Macro & RBI
Current Nifty and Bank Nifty levels (from live data), USD/INR rate, RBI stance, FII/DII flows.

## ⚡ Key Market Triggers This Week
3–5 bullets. Most important events driving markets. Be specific with numbers.

## ⚠️ Risks to Watch
2–3 bullets. Biggest near-term risks.

Keep each section to 3–4 sentences/bullets. No filler.

Output the markdown directly. Do not reproduce planning, instructions, or deliberation in the output.
"""
    result = call_ai(prompt, system=system, max_tokens=4096)
    return result if result.strip() else "Market intelligence could not be generated."

# =============================
# SECTOR AGGREGATION
# =============================

def build_sector_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        sector = SECTOR_MAP.get(r["Ticker"])
        if sector is None:
            continue
        rows.append({"Sector": sector, "Signal": r["Signal"],
                     "Ticker": r["Ticker"], "Upside %": r["Upside %"]})
    if not rows:
        return pd.DataFrame()
    agg     = pd.DataFrame(rows)
    summary = (
        agg.groupby("Sector")
        .agg(
            Breakouts   =("Signal",   lambda x: (x == "Breakout").sum()),
            Building    =("Signal",   lambda x: (x == "Building").sum()),
            Avg_Upside  =("Upside %", "mean"),
            Top_Tickers =("Ticker",   lambda x: ", ".join(x.tolist()[:4])),
        )
        .reset_index()
    )
    summary["Total Signals"] = summary["Breakouts"] + summary["Building"]
    summary["Avg Upside %"]  = summary["Avg_Upside"].round(1)
    return (
        summary[["Sector","Total Signals","Breakouts","Building","Avg Upside %","Top_Tickers"]]
        .sort_values("Total Signals", ascending=False)
        .reset_index(drop=True)
    )

def build_sector_context(sector_df: pd.DataFrame) -> str:
    if sector_df.empty:
        return "No named-sector data from scan."
    return "\n".join(
        f"- {r['Sector']}: {r['Total Signals']} signals "
        f"({r['Breakouts']} breakouts, {r['Building']} building) | "
        f"avg upside {r['Avg Upside %']}%"
        for _, r in sector_df.head(6).iterrows()
    )

def build_momentum_context(df: pd.DataFrame) -> str:
    if df.empty:
        return "No momentum candidates identified."
    return "\n".join(
        f"{r['Ticker']} ₹{r['Price ₹']} {r['Signal']} RSI{r['RSI']} "
        f"Vol{r['Vol Ratio']}x Target₹{r['Target ₹']}(+{r['Upside %']}%)"
        for _, r in df.head(20).iterrows()
    )

def condense_intel(intel_summary: str) -> str:
    """Extract only the key numbers/facts from intel summary for the strategy prompt.
    Keeps the strategy prompt small while preserving actionable macro context."""
    # Pull out lines containing numbers, percentages, named entities — skip prose
    key_lines = []
    for line in intel_summary.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Keep lines with prices, percentages, named indicators, or bullet content
        if any(c in line for c in ["₹", "$", "%", "•", "*", "-", "·"]) or \
           any(k in line.lower() for k in ["nifty","sensex","rbi","fii","dii","crude","brent","inr","usd","rate","inflation","gdp"]):
            key_lines.append(line.lstrip("•*- "))
    return "\n".join(key_lines[:12]) if key_lines else intel_summary[:400]

# =============================
# AI STRATEGY BRIEF
# - ETF live market data now passed in
# - Top 3 picks pre-filtered (RSI ≤ RSI_OVERBOUGHT) and explicitly
#   passed to AI with instruction: never flag them as risky/avoid
# =============================

def generate_strategy(
    intel_summary: str,
    momentum_context: str,
    sector_context: str,
    etf_context: str,
) -> str:
    macro_facts = condense_intel(intel_summary)

    # ── Ultra-compact contexts ──────────────────────────────────────────
    # ETF: top 8 only, single line each
    etf_lines = [l for l in etf_context.splitlines() if l.strip()][:8]
    etf_short  = "\n".join(etf_lines)

    # Stocks: top 12 only
    stock_lines = [l for l in momentum_context.splitlines() if l.strip()][:12]
    stocks_short = "\n".join(stock_lines)

    # Sectors: top 4 only
    sector_lines = [l for l in sector_context.splitlines() if l.strip()][:4]
    sector_short = "\n".join(sector_lines)

    system = "NSE equity analyst. Cite both technical AND macro reason per pick. Only use tickers from STOCKS. TOP 3 ≠ AVOID."

    prompt = f"""{now_ist().strftime('%d %b %Y')}
MACRO:{macro_facts}
ETF DATA (for ETF RADAR section only — DO NOT pick these as stocks):{etf_short}
SECTORS:{sector_short}
STOCK SCAN (use ONLY these tickers for TOP 3 BUY PICKS):{stocks_short}

Write markdown brief:
**MARKET PULSE** 2 sentences.

**TOP 3 BUY PICKS** — Pick ONLY from STOCK SCAN above. ETF tickers are forbidden here.
RSI≤{RSI_OVERBOUGHT}, breakout signal, macro tailwind. Why column: max 10 words, technical+macro.
One ticker per row, exactly 3 rows.
|Ticker|Entry ₹|Target ₹|Upside|Why|Stop ₹|Conviction|Horizon|
|------|-------|--------|------|---|------|----------|-------|

**ETF RADAR** — Pick ONLY from ETF DATA above. Top 5 by 1M momentum. One row each.
|ETF|Name|Trend|RSI|1M%|Action|
|---|----|----|---|---|------|

**WATCHLIST** 3 stocks from STOCK SCAN, one bullet each.
**SECTOR TO OWN** 1 sector+ETF ticker, 2 sentences.
**AVOID** 2 bullets, not Top 3 picks.
**TAIL RISKS** 2 bullets.

Output the markdown directly. Do not reproduce planning, instructions, or deliberation in the output."""

    # sarvam-105b is a reasoning model; at the starter-tier max_tokens cap
    # (4096) it usually empties its budget on chain-of-thought, so call_ai
    # falls back to salvaging the user-facing tail of reasoning_content.
    result = call_ai(prompt, system=system, max_tokens=4096)
    if result.strip():
        return result

    # Retry with absolute minimum prompt
    mini = f"""{now_ist().strftime('%d %b %Y')}
MACRO:{macro_facts[:300]}
STOCK SCAN (picks must come ONLY from here, no ETFs):{stocks_short[:400]}
Write: **MARKET PULSE** 2 sentences. **TOP 3 BUY PICKS** table from STOCK SCAN only. **AVOID** 2 bullets.

Output the markdown directly. Do not reproduce planning, instructions, or deliberation in the output."""
    result = call_ai(mini, system=system, max_tokens=4096)
    return result if result.strip() else "Strategy unavailable — API timeout. Scan data above is valid."

# =============================
# HELPERS
# =============================

def best_upside_picks(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Top N by composite Score (built in compute_composite_score). Excludes
    RSI > RSI_OVERBOUGHT and Stage 4 (extended). Falls back to all rows if
    everything is filtered out.
    """
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "Score" not in d.columns:
        d = compute_composite_score(d)
    candidates = d[(d.get("RSI", 0) <= RSI_OVERBOUGHT) & (d.get("StageId", 0) != 4)]
    if candidates.empty:
        candidates = d
    return (
        candidates.nlargest(n, "Score")[["Ticker", "Price ₹", "Target ₹", "Upside %", "Signal", "RSI", "Stage", "Score"]]
        .reset_index(drop=True)
    )

# =============================
# PIPELINE — pure function, no st.* calls. Called by both the foreground UI
# (with a widget-backed progress_cb) and the background thread (with a
# status-file-writing progress_cb).
# =============================

def _run_pipeline(cb=None) -> dict:
    def _step(phase, current, total, msg=""):
        if cb:
            try: cb(phase, current, total, msg)
            except Exception: pass

    _step("registry", 0, 100, "Loading NSE registry & Nifty benchmark")
    registry = get_nse_registry()
    if registry.empty:
        raise RuntimeError("NSE registry empty — cannot continue")
    nifty_close = get_nifty_close()

    _step("etf", 0, 100, "Scanning ETFs")
    etf_registry = get_etf_registry()
    etf_df = scan_etfs(etf_registry, progress_cb=cb) if not etf_registry.empty else pd.DataFrame()

    _step("scan", 0, 100, "Scanning equities")
    eq_symbols = registry["YF_SYMBOL"].tolist()
    momentum_df = scan_market(eq_symbols, progress_cb=cb, nifty_close=nifty_close)

    shortlist_df   = build_shortlist(momentum_df)
    early_rally_df = (
        momentum_df[momentum_df["StageId"].isin([1, 2])].sort_values("Score", ascending=False)
        if not momentum_df.empty else pd.DataFrame()
    )

    _step("intel", 60, 100, "Fetching live market snapshot + news")
    live_snapshot = get_live_market_snapshot()
    raw_news      = get_raw_news()

    _step("ai-intel", 70, 100, "AI market intelligence brief")
    intel_summary = generate_intel_summary(raw_news, live_snapshot)

    _step("ai-strategy", 80, 100, "AI strategy brief")
    sector_df        = build_sector_heatmap(momentum_df)
    sector_context   = build_sector_context(sector_df)
    momentum_context = build_momentum_context(momentum_df)
    etf_context      = build_etf_context(etf_df)
    strategy         = generate_strategy(intel_summary, momentum_context, sector_context, etf_context)

    _step("funds", 90, 100, "Deep-dive fundamentals (shortlist)")
    fund_map = {}
    if not shortlist_df.empty:
        sl_symbols = [f"{t}.NS" for t in shortlist_df["Ticker"].tolist()]
        raw_funds  = fetch_fundamentals_bulk(sl_symbols, progress_cb=cb)
        for _, t_row in shortlist_df.iterrows():
            sym  = f"{t_row['Ticker']}.NS"
            m    = raw_funds.get(sym, {})
            tech = t_row.to_dict()
            try:
                rec = recommend_action(m, tech)
            except Exception as e:
                rec = {"action": "HOLD", "conviction": 0, "scores": {},
                       "bull_case": [f"Recommender error: {e}"], "bear_case": [],
                       "entry_zone": (None, None), "stop_loss": None, "targets": {}}
            fund_map[sym] = {"metrics": m, "technicals": tech, "recommendation": rec}

    return {
        "momentum_df":    momentum_df,
        "shortlist_df":   shortlist_df,
        "early_rally_df": early_rally_df,
        "sector_df":      sector_df,
        "etf_df":         etf_df,
        "fund_map":       fund_map,
        "intel_summary":  intel_summary,
        "strategy":       strategy,
        "last_run":       now_ist(),
    }

# =============================
# RUN BUTTON
# =============================

# ── streamlit-autorefresh import (used while a background job is running) ──
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    def st_autorefresh(*args, **kwargs):  # graceful fallback if package missing
        return 0

# ── Drain any AI debug entries emitted from a background thread ──
if _AI_DEBUG_BUFFER:
    log = st.session_state.get("ai_debug", [])
    log.extend(_AI_DEBUG_BUFFER)
    st.session_state["ai_debug"] = log
    _AI_DEBUG_BUFFER.clear()

# ── Reattach to the latest completed background result, if any ──
_latest_mtime = _result_mtime()
if _latest_mtime > 0 and st.session_state.get("_result_mtime", 0) < _latest_mtime:
    _payload = _load_result()
    if _payload:
        st.session_state.update(_payload)
        st.session_state["_result_mtime"] = _latest_mtime

# ── Run button + status banner ────────────────────────────────────────
col_btn, col_note = st.columns([1, 3])
_job_running = _job_alive()
with col_btn:
    run = st.button(
        "⏳  Job running …" if _job_running else "⚡  Run Market Intelligence",
        disabled=_job_running,
    )
with col_note:
    st.markdown(
        "<small style='color:#4a5578;font-family:JetBrains Mono,monospace;font-size:0.7rem'>"
        "~2,000 NSE EQ stocks &nbsp;·&nbsp; 4-stage rally classifier &nbsp;·&nbsp; "
        "FMP-style fundamentals on shortlist &nbsp;·&nbsp; AI strategy &nbsp;·&nbsp; "
        "<strong style='color:#00c896'>safe to close tab once started</strong>"
        "</small>",
        unsafe_allow_html=True,
    )

if run and not _job_running:
    job_id = start_background_job()
    st.session_state["_active_job_id"] = job_id
    # Brief pause to let the thread write its first status entry
    time.sleep(0.4)
    st.rerun()

# ── Live status banner + auto-refresh while a job is running ──────────
if _job_alive():
    status = _read_status() or {}
    cur = int(status.get("current") or 0)
    tot = int(status.get("total") or 100) or 100
    pct = max(0, min(100, int(cur / tot * 100))) if tot else 0
    phase = status.get("phase", "—")
    msg   = status.get("msg", "")
    jid   = status.get("job_id", "—")
    started = status.get("started_at", "")
    try:
        started_h = datetime.fromisoformat(started).astimezone(IST).strftime("%H:%M IST")
    except Exception:
        started_h = "—"

    st.markdown(
        f"<div style='margin-top:14px;padding:14px 18px;border:1px solid rgba(212,168,67,0.25);"
        f"border-left:3px solid #d4a843;border-radius:0 12px 12px 0;background:rgba(15,20,40,0.7);"
        f"font-family:JetBrains Mono,monospace;font-size:0.8rem;color:#c0e8d8'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px'>"
        f"<span><strong style='color:#d4a843'>● BACKGROUND JOB</strong> &nbsp; "
        f"phase <strong>{phase}</strong> &nbsp; · &nbsp; {msg}</span>"
        f"<span style='color:#8896b3'>job <strong>{jid}</strong> · since {started_h}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )
    st.progress(pct)
    st.caption("This scan runs server-side — you can close this tab and the work will continue. The page will reload automatically every 3 s.")
    st_autorefresh(interval=3000, key="bg_job_poll")
elif (_read_status() or {}).get("error"):
    err = _read_status().get("error", "unknown error")
    st.error(f"Last background job failed: {err}")

# =============================
# DEEP DIVE PANEL — shared by the shortlist selectbox and the Search Stock tab
# =============================

def render_deep_dive_panel(ticker: str, tech_row: dict, metrics: dict, rec: dict,
                           context_label: str = "",
                           intel_summary: str = "") -> None:
    """Render the full per-stock deep dive UI. Used by both the shortlist
    selectbox path (Deep Dive tab) and the ad-hoc Search Stock tab."""
    m = metrics or {}

    # Header strip
    hcol1, hcol2, hcol3, hcol4 = st.columns([2, 1, 1, 1])
    sub_caption = m.get("name", "") or ""
    sector_line = f"{m.get('sector') or '—'} / {m.get('industry') or '—'}"
    meta_html = f"{sub_caption} &nbsp;·&nbsp; {sector_line}"
    if context_label:
        meta_html += f" &nbsp;·&nbsp; <span style='color:#d4a843'>{context_label}</span>"
    hcol1.markdown(
        f"<div class='ticker-name'>{ticker}</div>"
        f"<div class='ticker-meta'>{meta_html}</div>",
        unsafe_allow_html=True,
    )
    hcol2.metric("Price ₹", _num(tech_row.get("Price ₹")))
    hcol3.metric("Stage", tech_row.get("Stage", "—"))
    hcol4.metric("Score", _num(tech_row.get("Score"), decimals=1))

    # Recommendation card
    action = rec.get("action", "—")
    color_map = {"STRONG_BUY":"#00c896","BUY":"#00c896","ACCUMULATE":"#d4a843",
                 "HOLD":"#8896b3","AVOID":"#f87171"}
    ac_color = color_map.get(action, "#8896b3")
    st.markdown(f"""
    <div class="pick-card" style="margin-top:10px;border-left:3px solid {ac_color}">
        <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">
            <span class="upside-pill" style="background:{ac_color};color:#0a0a0a;font-size:0.95rem;padding:6px 18px">
                {action.replace('_',' ')}
            </span>
            <span style="font-family:JetBrains Mono,monospace;color:#8896b3">
                Conviction <strong style="color:#f0c96a;font-size:1.1rem">{rec.get('conviction','—')}</strong> / 100
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if rec.get("conviction") is not None:
        st.progress(min(int(rec["conviction"]), 100))

    scores = rec.get("scores", {})
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Tech",    f"{scores.get('technical','—')}")
    sc2.metric("Quality", f"{scores.get('quality','—')}")
    sc3.metric("Value",   f"{scores.get('value','—')}")
    sc4.metric("Growth",  f"{scores.get('growth','—')}")

    # Sub-tabs for fundamentals
    stab_v, stab_q, stab_g, stab_o, stab_th = st.tabs(
        ["Valuation", "Quality", "Growth", "Ownership", "Bull/Bear & Targets"]
    )

    with stab_v:
        dcf = m.get("dcf") or {}
        v1, v2, v3, v4 = st.columns(4)
        v1.metric("P/E trailing", _num(m.get("pe_trailing")))
        v2.metric("P/E forward",  _num(m.get("pe_forward")))
        v3.metric("P/B",          _num(m.get("pb")))
        v4.metric("EV/EBITDA",    _num(m.get("ev_ebitda")))
        v5, v6, v7, v8 = st.columns(4)
        v5.metric("PEG",          _num(m.get("peg")))
        v6.metric("P/S",          _num(m.get("price_to_sales")))
        v7.metric("Div Yield",    _pct(m.get("dividend_yield")))
        v8.metric("DCF Intrinsic ₹", _num(dcf.get("intrinsic_per_share")))
        d1, d2, d3 = st.columns(3)
        d1.metric("DCF Upside %", _num(dcf.get("upside_pct"), suffix="%"))
        d2.metric("DCF Confidence", dcf.get("confidence", "—"))
        a = dcf.get("assumptions", {})
        d3.metric("Growth used", _pct(a.get("growth")) if a else "—")
        if a:
            st.caption(
                f"DCF: 5y FCF @ growth {_pct(a.get('growth'))} · discount {_pct(a.get('discount_rate'))} · "
                f"terminal {_pct(a.get('terminal_growth'))} · history {a.get('fcf_history_years','—')}y"
            )

    with stab_q:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("ROE",   _pct(m.get("roe")))
        q2.metric("ROCE",  _pct(m.get("roce")))
        q3.metric("ROIC",  _pct(m.get("roic")))
        q4.metric("ROA",   _pct(m.get("roa")))
        q5, q6, q7, q8 = st.columns(4)
        q5.metric("Gross Margin", _pct(m.get("gross_margin")))
        q6.metric("Op Margin",    _pct(m.get("op_margin")))
        q7.metric("Net Margin",   _pct(m.get("net_margin")))
        q8.metric("D/E",          _num(m.get("debt_to_equity")))
        q9, q10, q11, _ = st.columns(4)
        q9.metric("Interest Cov", _num(m.get("interest_coverage"), suffix="×"))
        q10.metric("Current Ratio", _num(m.get("current_ratio")))
        q11.metric("Quick Ratio",   _num(m.get("quick_ratio")))

    with stab_g:
        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Rev CAGR 3Y",   _pct(m.get("rev_cagr_3y")))
        g2.metric("Rev CAGR 5Y",   _pct(m.get("rev_cagr_5y")))
        g3.metric("EPS CAGR 3Y",   _pct(m.get("eps_cagr_3y")))
        g4.metric("Fwd EPS Growth", _pct(m.get("fwd_eps_growth")))
        g5, g6, g7, _ = st.columns(4)
        g5.metric("Revenue Growth (TTM)",  _pct(m.get("revenue_growth")))
        g6.metric("Earnings Growth (TTM)", _pct(m.get("earnings_growth")))
        g7.metric("Avg Earnings Surprise", _num(m.get("earnings_surprise_avg"), suffix="%"))

    with stab_o:
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Insider Held",       _pct(m.get("insider_held")))
        o2.metric("Institutional Held", _pct(m.get("institutional_held")))
        o3.metric("Analyst Rec",        m.get("recommendation_key") or "—")
        o4.metric("Analyst Count",      _num(m.get("analyst_count"), decimals=0))
        t1, t2, t3 = st.columns(3)
        t1.metric("Analyst Target Mean ₹", _num(m.get("target_mean")))
        t2.metric("Analyst Target High ₹", _num(m.get("target_high")))
        t3.metric("Analyst Target Low ₹",  _num(m.get("target_low")))
        hl = m.get("news_headlines") or []
        if hl:
            st.markdown("**Recent news headlines:**")
            for h in hl:
                st.markdown(f"- {h}")

    with stab_th:
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("**Bull case**")
            for b in rec.get("bull_case", []):
                st.markdown(f"- {b}")
        with b2:
            st.markdown("**Bear case**")
            for b in rec.get("bear_case", []):
                st.markdown(f"- {b}")
        ez = rec.get("entry_zone", (None, None))
        e1, e2, e3 = st.columns(3)
        e1.metric("Entry Zone ₹", f"{_num(ez[0])} – {_num(ez[1])}")
        e2.metric("Stop Loss ₹",  _num(rec.get("stop_loss")))
        tgts = rec.get("targets", {})
        e3.metric("Targets count", f"{len(tgts)}")
        if tgts:
            price_val = tech_row.get("Price ₹")
            tdf = pd.DataFrame(
                [(k, v, ((v / price_val - 1) * 100 if price_val else None))
                 for k, v in tgts.items()],
                columns=["Source", "Target ₹", "Upside %"],
            ).sort_values("Target ₹")
            st.dataframe(
                tdf.style.format({"Target ₹": "₹{:.2f}", "Upside %": "{:+.1f}%"}, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

        if st.button("Generate AI thesis for this stock", key=f"ai_thesis_{ticker}"):
            sys = "NSE equity analyst. 4 sentences max. Cite numbers and macro context."
            macro_block = ""
            if intel_summary:
                macro_short = condense_intel(intel_summary)[:1200]
                if macro_short:
                    macro_block = f"\n\nCURRENT MACRO CONTEXT:\n{macro_short}\n"
            prompt_th = (
                f"{ticker} ({m.get('name','')}): price ₹{tech_row.get('Price ₹')}, "
                f"Stage {tech_row.get('Stage')}, RSI {tech_row.get('RSI')}, ADX {tech_row.get('ADX')}, "
                f"RS {tech_row.get('RS vs Nifty')}. ROE { _pct(m.get('roe')) }, "
                f"Op margin { _pct(m.get('op_margin')) }, P/E fwd { _num(m.get('pe_forward')) }, "
                f"DCF upside { _num((m.get('dcf') or {}).get('upside_pct'), suffix='%') }. "
                f"Action: {rec.get('action')}, conviction {rec.get('conviction')}."
                f"{macro_block}\n"
                f"Write a 4-sentence investment thesis covering: 1) technical setup, "
                f"2) fundamentals, 3) valuation, 4) how the current macro context "
                f"(Nifty / USD-INR / Brent / RBI / FII-DII flows) affects this name. "
                f"Output the thesis directly. Do not reproduce planning, instructions, or deliberation."
            )
            with st.spinner("Drafting thesis …"):
                thesis = call_ai(prompt_th, system=sys, max_tokens=4096)
            if thesis:
                st.markdown(f'<div class="intel-wrap">{thesis}</div>', unsafe_allow_html=True)
            else:
                st.warning("Thesis unavailable (AI call failed). Check Debug Log.")

# =============================
# DISPLAY RESULTS
# =============================

if "strategy" in st.session_state:
    momentum_df    = st.session_state["momentum_df"]
    shortlist_df   = st.session_state.get("shortlist_df", pd.DataFrame())
    early_rally_df = st.session_state.get("early_rally_df", pd.DataFrame())
    sector_df_st   = st.session_state.get("sector_df", pd.DataFrame())
    fund_map       = st.session_state.get("fund_map", {})
    intel_summary  = st.session_state.get("intel_summary", "")
    strategy       = st.session_state["strategy"]
    etf_df         = st.session_state.get("etf_df", pd.DataFrame())

    # ── KPI Row (global, above tabs) ──
    if not momentum_df.empty:
        k1, k2, k3, k4, k5 = st.columns(5)
        breakouts = int((momentum_df["StageId"] == 3).sum())
        early     = int(momentum_df["StageId"].isin([1, 2]).sum())
        avg_score = float(momentum_df["Score"].mean())
        k1.metric("Signals",          f"{len(momentum_df)}")
        k2.metric("Early Rally",      f"{early}", f"+{int((momentum_df['StageId']==2).sum())} stage 2")
        k3.metric("Breakouts",        f"{breakouts}")
        k4.metric("Avg Composite",    f"{avg_score:.1f}")
        k5.metric("Best Upside",      f"{momentum_df['Upside %'].max():.1f}%")

    tab_scan, tab_early, tab_deep, tab_search, tab_sector, tab_intel, tab_dl = st.tabs([
        "Momentum Scan", "Early Rally", "Deep Dive", "Search Stock",
        "Sector Heatmap", "Intel & Strategy", "Downloads",
    ])

    # =======================================================
    # TAB 1 — MOMENTUM SCAN
    # =======================================================
    with tab_scan:
        if not momentum_df.empty:
            st.markdown('<div class="section-header">Top Composite Picks</div>', unsafe_allow_html=True)
            st.caption("Top 3 by composite Score (stage + MACD + ADX + RS + RSI + upside). Excludes RSI overbought and Stage 4 extended.")
            top3 = best_upside_picks(momentum_df, n=3)
            cols = st.columns(3)
            for idx, (_, row) in enumerate(top3.iterrows()):
                sig_cls = "sig-breakout" if row["Signal"] == "Breakout" else "sig-building"
                with cols[idx]:
                    st.markdown(f"""
                    <div class="pick-card">
                        <div class="ticker-name">{row['Ticker']}</div>
                        <div class="ticker-meta">₹{row['Price ₹']} &nbsp;·&nbsp; RSI {row['RSI']} &nbsp;·&nbsp; Stage: {row['Stage']}</div>
                        <span class="upside-pill">▲ {row['Upside %']}% &nbsp; Score {row['Score']:.0f}</span>
                        <div class="target-row">Target &nbsp;<span class="target-price">₹{row['Target ₹']}</span></div>
                        <div class="signal-tag {sig_cls}">{row['Signal'].upper()}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('<div class="section-header">Full Momentum Candidates</div>', unsafe_allow_html=True)

            def color_stage(val):
                colors = {"Accumulation":"#4f8ef7","Early Markup":"#d4a843",
                          "Breakout":"#00c896","Extended":"#f87171","—":"#4a5578"}
                c = colors.get(val, "")
                return f"color:{c};font-weight:700" if c else ""
            def color_score(val):
                if val >= 75: return "color:#00c896;font-weight:700"
                if val >= 55: return "color:#d4a843;font-weight:600"
                return ""
            def color_upside(val):
                if val >= 20: return "color:#00c896;font-weight:700"
                if val >= 10: return "color:#d4a843"
                return ""
            def color_rsi_stock(val):
                return "color:#f87171;font-weight:700" if val and val > RSI_OVERBOUGHT else ""

            display_cols = [c for c in [
                "Ticker","Price ₹","Stage","Score","Signal","RSI","Vol Ratio",
                "ADX","MACD Hist","RS vs Nifty","ROC20 %","MA50","MA200",
                "Dist MA50 %","Dist MA200 %","Target ₹","Upside %","52W High ₹","Gap to 52W %",
            ] if c in momentum_df.columns]

            fmt = {
                "Price ₹":"₹{:.2f}","MA50":"₹{:.2f}","MA200":"₹{:.2f}","Target ₹":"₹{:.2f}",
                "52W High ₹":"₹{:.2f}","RSI":"{:.1f}","Vol Ratio":"{:.2f}×",
                "Upside %":"+{:.1f}%","Gap to 52W %":"{:.1f}%","ADX":"{:.1f}",
                "MACD Hist":"{:.3f}","RS vs Nifty":"{:.2f}","ROC20 %":"{:+.1f}%",
                "Dist MA50 %":"{:+.1f}%","Dist MA200 %":"{:+.1f}%","Score":"{:.1f}",
            }
            fmt = {k: v for k, v in fmt.items() if k in display_cols}

            styler = momentum_df[display_cols].style.format(fmt, na_rep="N/A")
            if "Stage"    in display_cols: styler = styler.map(color_stage,      subset=["Stage"])
            if "Score"    in display_cols: styler = styler.map(color_score,      subset=["Score"])
            if "Upside %" in display_cols: styler = styler.map(color_upside,     subset=["Upside %"])
            if "RSI"      in display_cols: styler = styler.map(color_rsi_stock,  subset=["RSI"])

            st.dataframe(styler, use_container_width=True, height=520)
            st.caption(f"All {len(momentum_df)} signals · ranked by composite Score. Stage 4 rows are kept but down-weighted.")
        else:
            st.info("No momentum candidates found in this scan.")

    # =======================================================
    # TAB 2 — EARLY RALLY
    # =======================================================
    with tab_early:
        st.markdown('<div class="section-header">Early Rally Radar · Stage 1 + 2</div>', unsafe_allow_html=True)
        st.caption("Stocks set up BEFORE the breakout. Stage 1 = accumulation (squeeze, OBV rising). Stage 2 = early markup (MACD up, ADX rising, above MA50>MA200).")
        if early_rally_df is None or early_rally_df.empty:
            st.info("No Stage 1 or Stage 2 setups detected in this scan.")
        else:
            # Card grid, 3 per row
            rows = early_rally_df.head(18).to_dict("records")
            for i in range(0, len(rows), 3):
                cols = st.columns(3)
                for j, row in enumerate(rows[i:i + 3]):
                    stage_color = "#4f8ef7" if row["StageId"] == 1 else "#d4a843"
                    why = []
                    if row.get("ADX") and row["ADX"] >= 18:               why.append(f"ADX {row['ADX']:.0f}")
                    if row.get("MACD Hist") and row["MACD Hist"] > 0:     why.append("MACD bullish")
                    if row.get("RS vs Nifty") and row["RS vs Nifty"] > 1: why.append(f"RS {row['RS vs Nifty']:.2f}")
                    if row.get("Vol Ratio") and row["Vol Ratio"] > 1.2:   why.append(f"Vol {row['Vol Ratio']:.1f}×")
                    why_text = " · ".join(why) if why else "Setup forming"
                    with cols[j]:
                        st.markdown(f"""
                        <div class="pick-card" style="border-left:2px solid {stage_color};">
                            <div class="ticker-name">{row['Ticker']}</div>
                            <div class="ticker-meta">₹{row['Price ₹']} &nbsp;·&nbsp; RSI {row['RSI']} &nbsp;·&nbsp;
                                <span style="color:{stage_color};font-weight:700">Stage {row['StageId']} · {row['Stage']}</span>
                            </div>
                            <span class="upside-pill">Score {row['Score']:.0f} &nbsp; ▲ {row['Upside %']}%</span>
                            <div class="target-row" style="margin-top:10px;font-size:0.74rem;color:#8896b3">
                                {why_text}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            st.caption(f"{len(early_rally_df)} early-stage setups detected. Sorted by composite Score.")

    # =======================================================
    # TAB 3 — DEEP DIVE
    # =======================================================
    with tab_deep:
        st.markdown('<div class="section-header">Per-Stock Deep Dive · Fundamentals + Recommendation</div>', unsafe_allow_html=True)
        if shortlist_df is None or shortlist_df.empty:
            st.info("No shortlist available. Run the scan to generate a shortlist of top-scored stocks.")
        else:
            if len(shortlist_df) <= SHORTLIST_FLOOR:
                st.warning(f"Only {len(shortlist_df)} stocks passed the score threshold ({SHORTLIST_MIN_SCORE}). Showing the top {len(shortlist_df)} by score.")
            tickers = shortlist_df["Ticker"].tolist()
            selected_ticker = st.selectbox("Select a shortlisted stock", tickers, key="deepdive_ticker")
            yf_sym  = f"{selected_ticker}.NS"
            payload = fund_map.get(yf_sym, {})
            m       = payload.get("metrics", {}) if payload else {}
            rec     = payload.get("recommendation", {}) if payload else {}
            tech_row = shortlist_df[shortlist_df["Ticker"] == selected_ticker].iloc[0].to_dict()
            score_v  = tech_row.get("Score")
            ctx_label = f"From shortlist · score {score_v:.0f}" if score_v is not None else "From shortlist"
            render_deep_dive_panel(
                selected_ticker, tech_row, m, rec,
                context_label=ctx_label, intel_summary=intel_summary,
            )

    # =======================================================
    # TAB 4 — SEARCH STOCK (ad-hoc deep dive on any NSE EQ symbol)
    # =======================================================
    with tab_search:
        st.markdown('<div class="section-header">Search Any NSE Stock · Ad-hoc Deep Dive</div>', unsafe_allow_html=True)
        st.caption("Type any NSE EQ ticker or company name. Same indicators, fundamentals, DCF, and recommendation that the shortlist uses — computed on demand and cached for 1 hour.")

        registry = get_nse_registry()
        if registry is None or registry.empty:
            st.warning("NSE registry unavailable — try again in a moment.")
        else:
            reg = registry.copy()
            name_col = next((c for c in reg.columns if "NAME" in c), None)
            if name_col:
                reg["label"] = reg["SYMBOL"].astype(str) + " — " + reg[name_col].astype(str).str.strip()
            else:
                reg["label"] = reg["SYMBOL"].astype(str)
            labels = reg["label"].tolist()
            label_to_symbol = dict(zip(reg["label"], reg["SYMBOL"].astype(str)))

            picked_label = st.selectbox(
                "Search any NSE EQ stock",
                labels, index=None,
                placeholder="Start typing ticker or company name…",
                key="search_stock_label",
            )

            if picked_label:
                picked_symbol = label_to_symbol.get(picked_label, "").strip().upper()
                yf_sym = f"{picked_symbol}.NS"
                with st.spinner(f"Computing technicals + fundamentals for {picked_symbol} …"):
                    tech_row = compute_single_stock_technicals(yf_sym)
                    metrics  = fetch_fundamentals(yf_sym) if tech_row is not None else {}
                if tech_row is None:
                    st.warning(f"No usable price data for {picked_symbol}. The symbol may be delisted, suspended, or too thinly traded.")
                else:
                    try:
                        rec = recommend_action(metrics, tech_row)
                    except Exception as e:
                        rec = {"action": "HOLD", "conviction": 0, "scores": {},
                               "bull_case": [f"Recommender error: {e}"], "bear_case": [],
                               "entry_zone": (None, None), "stop_loss": None, "targets": {}}
                    in_shortlist = (shortlist_df is not None and not shortlist_df.empty
                                    and picked_symbol in set(shortlist_df["Ticker"].astype(str)))
                    ctx_label = "Ad-hoc lookup · in shortlist" if in_shortlist else "Ad-hoc lookup · not in shortlist"
                    render_deep_dive_panel(
                        picked_symbol, tech_row, metrics, rec,
                        context_label=ctx_label, intel_summary=intel_summary,
                    )
            else:
                st.info("Select a stock from the dropdown above to see its full deep dive.")

    # =======================================================
    # TAB 4 — SECTOR HEATMAP
    # =======================================================
    with tab_sector:
        st.markdown('<div class="section-header">Sector Heatmap</div>', unsafe_allow_html=True)
        if sector_df_st is None or sector_df_st.empty:
            st.info("No mapped-sector data available from this scan.")
        else:
            def color_avg_up(val):
                if val is None or not np.isfinite(val): return ""
                if val >= 20: return "color:#00c896;font-weight:700"
                if val >= 10: return "color:#d4a843"
                return ""
            st.dataframe(
                sector_df_st.style
                    .map(color_avg_up, subset=["Avg Upside %"])
                    .format({"Avg Upside %": "+{:.1f}%"}),
                use_container_width=True, height=460,
            )

    # =======================================================
    # TAB 5 — INTEL & STRATEGY
    # =======================================================
    with tab_intel:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            st.markdown('<div class="section-header">AI Strategy Brief</div>', unsafe_allow_html=True)
            if strategy:
                st.markdown(f'<div class="strategy-wrap">\n\n{strategy}\n\n</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠ Strategy brief could not be generated. Check the AI Debug Log below.")
        with col_r:
            st.markdown('<div class="section-header">Market Intelligence</div>', unsafe_allow_html=True)
            scan_time = st.session_state["last_run"].strftime("%d %b %Y · %H:%M IST")
            st.markdown(
                f'<div class="intel-wrap">'
                f'<div class="intel-scan-time">AI Analysis &nbsp;·&nbsp; {scan_time}</div>'
                f'\n\n{intel_summary if intel_summary else ""}\n\n'
                f'</div>',
                unsafe_allow_html=True,
            )

    # =======================================================
    # TAB 6 — DOWNLOADS
    # =======================================================
    with tab_dl:
        st.markdown('<div class="section-header">Downloads</div>', unsafe_allow_html=True)
        stamp = st.session_state["last_run"].strftime("%Y%m%d_%H%M_IST")

        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Full scan (CSV)**")
            st.caption(f"All {len(momentum_df)} momentum signals with all indicators and Score.")
            if not momentum_df.empty:
                st.download_button(
                    label="⬇ Download full scan CSV",
                    data=momentum_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"nse_scan_{stamp}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.info("Nothing to download yet.")

        with d2:
            st.markdown("**Shortlist deep-dive (Excel)**")
            st.caption(f"{len(shortlist_df)} stocks — Summary, Valuation, Quality, Growth, Ownership, Recommendation + per-stock sheets.")
            if shortlist_df is not None and not shortlist_df.empty:
                try:
                    xlsx_bytes = build_excel_workbook(momentum_df, shortlist_df, fund_map)
                    st.download_button(
                        label="⬇ Download deep-dive XLSX",
                        data=xlsx_bytes,
                        file_name=f"nse_deepdive_{stamp}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Excel build failed: {e}")
            else:
                st.info("Run scan first.")

        if not shortlist_df.empty:
            st.markdown('<div class="section-header" style="margin-top:30px">Shortlist preview</div>', unsafe_allow_html=True)
            preview_cols = [c for c in ["Ticker","Price ₹","Stage","Score","RSI","ADX","RS vs Nifty","Upside %","Target ₹"] if c in shortlist_df.columns]
            st.dataframe(
                shortlist_df[preview_cols].style.format({
                    "Price ₹":"₹{:.2f}","Target ₹":"₹{:.2f}","RSI":"{:.1f}",
                    "ADX":"{:.1f}","RS vs Nifty":"{:.2f}","Score":"{:.1f}","Upside %":"+{:.1f}%",
                }),
                use_container_width=True, height=420,
            )

    # ── Debug Log ──
    if st.session_state.get("ai_debug"):
        with st.expander("🔧 AI Debug Log"):
            st.json(st.session_state["ai_debug"])

    st.markdown(
        f"<div class='nse-footer'>"
        f"Last scan &nbsp;·&nbsp; {st.session_state['last_run'].strftime('%H:%M:%S IST')}"
        f" &nbsp;·&nbsp; NSE Alpha AI"
        f"</div>",
        unsafe_allow_html=True,
    )

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-glyph">◈</div>
        <div class="empty-title">
            Hit &nbsp;<strong style="color:#d4a843">Run Market Intelligence</strong>&nbsp; to begin
        </div>
        <div class="empty-sub">
            ~2,000 NSE EQ stocks &nbsp;·&nbsp; 4-stage rally detector &nbsp;·&nbsp;
            FMP-style fundamentals &nbsp;·&nbsp; deep-dive recommendations &nbsp;·&nbsp;
            CSV + Excel download
        </div>
    </div>
    """, unsafe_allow_html=True)
