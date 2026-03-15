from **future** import annotations
import re
import streamlit as st
import requests
import io
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from zoneinfo import ZoneInfo
from tavily import TavilyClient

IST = ZoneInfo(“Asia/Kolkata”)

# =============================

# API CONFIG

# =============================

SARVAM_API_KEY = st.secrets[“SARVAM_API_KEY”]
TAVILY_API_KEY = st.secrets[“TAVILY_API_KEY”]
SARVAM_URL     = “https://api.sarvam.ai/v1/chat/completions”

tavily = TavilyClient(api_key=TAVILY_API_KEY)

# =============================

# SCANNER CONSTANTS

# =============================

MA_PERIOD          = 50
BREAKOUT_LOOKBACK  = 20
VOL_LOOKBACK       = 20
VOL_SURGE_THRESH   = 1.8
BREAKOUT_TOLERANCE = 0.99
MIN_BARS           = 60
CHUNK_SIZE         = 80
MAX_DISPLAY_STOCKS = 12
RSI_PERIOD         = 14
RSI_OVERBOUGHT     = 72   # stocks above this are excluded from Top 3 picks

# =============================

# ETF REGISTRY — fetched live from NSE, zero hardcoding

# Tries the NSE ETF CSV endpoint; yfinance symbol = SYMBOL + “.NS”

# =============================

@st.cache_data(ttl=86400)
def get_etf_registry() -> pd.DataFrame:
“””
Download NSE ETF list. Aggressively filter to only genuine ETF symbols:
- No leading digits
- Alphanumeric only (no special chars)
- Symbol length 3–20 chars
- Series == ‘EQ’ if column present
- Must contain ETF-like keyword in symbol OR name
“””
headers = {
“User-Agent”: “Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36”,
“Accept”    : “text/html,application/xhtml+xml,*/*;q=0.8”,
“Referer”   : “https://www.nseindia.com/”,
}
candidates = [
“https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv”,
“https://nsearchives.nseindia.com/content/ETF/eq_etfseclist.csv”,
“https://nsearchives.nseindia.com/content/equities/etf_seclist.csv”,
]

```
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
```

SECTOR_MAP = {
“HDFCBANK”:“Banking”,“ICICIBANK”:“Banking”,“SBIN”:“Banking”,“AXISBANK”:“Banking”,
“KOTAKBANK”:“Banking”,“INDUSINDBK”:“Banking”,“BANDHANBNK”:“Banking”,“FEDERALBNK”:“Banking”,
“IDFCFIRSTB”:“Banking”,“PNB”:“Banking”,“BANKBARODA”:“Banking”,“CANBK”:“Banking”,
“BAJFINANCE”:“Finance”,“BAJAJFINSV”:“Finance”,“CHOLAFIN”:“Finance”,“MUTHOOTFIN”:“Finance”,
“HDFC”:“Finance”,“RECLTD”:“Finance”,“PFC”:“Finance”,“IRFC”:“Finance”,
“TCS”:“IT”,“INFY”:“IT”,“WIPRO”:“IT”,“HCLTECH”:“IT”,“TECHM”:“IT”,
“LTIM”:“IT”,“MPHASIS”:“IT”,“PERSISTENT”:“IT”,“COFORGE”:“IT”,“OFSS”:“IT”,
“SUNPHARMA”:“Pharma”,“DRREDDY”:“Pharma”,“CIPLA”:“Pharma”,“DIVISLAB”:“Pharma”,
“APOLLOHOSP”:“Healthcare”,“MAXHEALTH”:“Healthcare”,“FORTIS”:“Healthcare”,
“LUPIN”:“Pharma”,“TORNTPHARM”:“Pharma”,“AUROPHARMA”:“Pharma”,“ALKEM”:“Pharma”,
“HEROMOTOCO”:“Auto”,“EICHERMOT”:“Auto”,“TVSMOTOR”:“Auto”,“ASHOKLEY”:“Auto”,
“MOTHERSON”:“Auto”,“BOSCHLTD”:“Auto”,“BHARATFORG”:“Auto”,“BALKRISIND”:“Auto”,
“TATASTEEL”:“Metals”,“JSWSTEEL”:“Metals”,“HINDALCO”:“Metals”,“VEDL”:“Metals”,
“SAIL”:“Metals”,“NMDC”:“Metals”,“COALINDIA”:“Metals”,“NATIONALUM”:“Metals”,
“RELIANCE”:“Energy”,“ONGC”:“Energy”,“IOC”:“Energy”,“BPCL”:“Energy”,
“GAIL”:“Energy”,“POWERGRID”:“Energy”,“NTPC”:“Energy”,“ADANIGREEN”:“Energy”,
“TATAPOWER”:“Energy”,“TORNTPOWER”:“Energy”,“CESC”:“Energy”,
“HINDUNILVR”:“FMCG”,“ITC”:“FMCG”,“NESTLEIND”:“FMCG”,“BRITANNIA”:“FMCG”,
“DABUR”:“FMCG”,“MARICO”:“FMCG”,“COLPAL”:“FMCG”,“EMAMILTD”:“FMCG”,
“LT”:“Infra”,“ADANIPORTS”:“Infra”,“SIEMENS”:“CapGoods”,“ABB”:“CapGoods”,
“BHEL”:“CapGoods”,“HAL”:“Defence”,“BEL”:“Defence”,“COCHINSHIP”:“Defence”,
“DLF”:“Realty”,“GODREJPROP”:“Realty”,“OBEROIRLTY”:“Realty”,“PHOENIXLTD”:“Realty”,
“PRESTIGE”:“Realty”,“BRIGADE”:“Realty”,
“BHARTIARTL”:“Telecom”,“IDEA”:“Telecom”,“INDUSTOWER”:“Telecom”,
“PIDILITIND”:“Chemicals”,“SRF”:“Chemicals”,“DEEPAKNITR”:“Chemicals”,
“NAVINFLUOR”:“Chemicals”,“ATUL”:“Chemicals”,
}

# =============================

# PAGE CONFIG

# =============================

st.set_page_config(page_title=“NSE Alpha AI”, layout=“wide”, initial_sidebar_state=“collapsed”)

st.markdown(”””

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

“””, unsafe_allow_html=True)

# =============================

# HEADER

# =============================

st.markdown(f”””

<div class="nse-header">
    <div class="nse-logo-row">
        <span class="nse-wordmark">NSE Alpha</span>
        <span class="nse-badge">AI</span>
    </div>
    <div class="nse-tagline">
        Market Intelligence &nbsp;·&nbsp;
        <em>{datetime.now(IST).strftime("%d %b %Y")}</em> &nbsp;·&nbsp;
        <em>{datetime.now(IST).strftime("%H:%M IST")}</em> &nbsp;·&nbsp;
        <em>Equity Momentum Scanner</em>
    </div>
    <div class="header-divider"></div>
</div>
""", unsafe_allow_html=True)

# =============================

# AI CALL

# =============================

def _strip_think(text: str) -> str:
return re.sub(r”<think>.*?</think>”, “”, text, flags=re.DOTALL).strip()

def call_ai(prompt: str, system: str = “”, max_tokens: int = 6000) -> str:
headers  = {“Authorization”: f”Bearer {SARVAM_API_KEY}”, “Content-Type”: “application/json”}
messages = []
if system:
messages.append({“role”: “system”, “content”: system})
messages.append({“role”: “user”, “content”: prompt})

```
debug_log = st.session_state.get("ai_debug", [])

payload = {
    "model"      : "sarvam-105b",
    "messages"   : messages,
    "max_tokens" : max_tokens,
    "temperature": 0.5,
    "top_p"      : 1,
}

try:
    r = requests.post(SARVAM_URL, headers=headers, json=payload, timeout=240)
    entry = {"model": "sarvam-105b", "status": r.status_code, "body": r.text[:1200]}
    debug_log.append(entry)
    st.session_state["ai_debug"] = debug_log

    if r.status_code == 200:
        choices = r.json().get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            stripped = _strip_think(str(content))
            if stripped:
                return stripped
except requests.exceptions.Timeout:
    debug_log.append({"model": "sarvam-105b", "error": "Timed out after 240s"})
    st.session_state["ai_debug"] = debug_log
except Exception as e:
    debug_log.append({"model": "sarvam-105b", "error": str(e)})
    st.session_state["ai_debug"] = debug_log

return ""
```

# =============================

# NSE REGISTRY

# =============================

@st.cache_data(ttl=86400)
def get_nse_registry() -> pd.DataFrame:
url     = “https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv”
headers = {“User-Agent”: “Mozilla/5.0”}
try:
r = requests.get(url, headers=headers, timeout=20)
r.raise_for_status()
df = pd.read_csv(io.StringIO(r.text))
df.columns = df.columns.str.strip().str.upper()
missing = {“SERIES”, “SYMBOL”} - set(df.columns)
if missing:
st.error(f”NSE format changed. Missing: {missing}”)
return pd.DataFrame()
df = df[df[“SERIES”] == “EQ”].copy()
df[“YF_SYMBOL”] = df[“SYMBOL”].astype(str).str.strip() + “.NS”
return df.reset_index(drop=True)
except Exception as e:
st.error(f”NSE registry download failed: {e}”)
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
high, low, cp = df[“High”], df[“Low”], df[“Close”].shift(1)
tr  = pd.concat([(high - low), (high - cp).abs(), (low - cp).abs()], axis=1).max(axis=1)
atr = tr.rolling(period).mean().iloc[-1]
price = df[“Close”].iloc[-1]
return round(float(atr / price * 100), 2) if price else np.nan

def estimate_upside(df: pd.DataFrame, price: float, high20: float) -> dict:
high52 = df[“High”].iloc[-252:].max() if len(df) >= 252 else df[“High”].max()
low52  = df[“Low”].iloc[-252:].min()  if len(df) >= 252 else df[“Low”].min()
atr_target = price + 2 * (compute_atr_pct(df) / 100 * price)
fib_target = low52 + (high52 - low52) * 1.618
targets    = [t for t in [high52, atr_target, fib_target] if t > price * 1.02]
target     = round(min(targets), 2) if targets else round(price * 1.12, 2)
return {
“target”        : target,
“upside_pct”    : round((target - price) / price * 100, 1),
“high52”        : round(high52, 2),
“high52_gap_pct”: round((high52 - price) / price * 100, 1),
}

# =============================

# ETF SCANNER — dynamic registry, sanity-checked data, live progress

# =============================

def scan_etfs(etf_registry: pd.DataFrame, _counter: st.delta_generator.DeltaGenerator | None = None, _pbar: st.delta_generator.DeltaGenerator | None = None) -> pd.DataFrame:
“”“Fetch live price, RSI, trend, returns for every ETF. Shows live count.”””
if etf_registry.empty:
return pd.DataFrame()

```
tickers  = etf_registry["YF_SYMBOL"].tolist()
name_map = dict(zip(etf_registry["YF_SYMBOL"], etf_registry["NAME"]))
results  = []
total    = len(tickers)

for i in range(0, total, 50):
    chunk = tickers[i : i + 50]
    done  = min(i + 50, total)
    if _pbar:
        _pbar.progress(int(done / total * 100))
    if _counter:
        _counter.markdown(
            f"<span style='font-family:JetBrains Mono,monospace;font-size:0.76rem;color:#8896b3'>"
            f"ETFs &nbsp;<strong style='color:#00c896'>{done}</strong> / {total}</span>",
            unsafe_allow_html=True,
        )
    try:
        data = yf.download(
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
```

def build_etf_context(etf_df: pd.DataFrame) -> str:
if etf_df.empty:
return “ETF data unavailable.”
# Only ETFs with valid price > ₹1 and valid 1M return; top 15 by absolute 1M momentum
valid = etf_df[
(etf_df[“Price ₹”] > 1) &
(etf_df[“1M %”].notna()) &
(etf_df[“RSI”].notna())
].copy()
if valid.empty:
valid = etf_df
top = valid.reindex(valid[“1M %”].abs().sort_values(ascending=False).index).head(15)
lines = []
for _, r in top.iterrows():
rsi_flag = “[OB]” if r[“RSI”] > RSI_OVERBOUGHT else (”[OS]” if r[“RSI”] < 35 else “”)
lines.append(
f”{r[‘ETF’]} ({r[‘Name’][:30]}): ₹{r[‘Price ₹’]} RSI{r[‘RSI’]:.0f}{rsi_flag} “
f”{r[‘Trend’].split()[0]} 1M{r[‘1M %’]:+.1f}% 3M{r[‘3M %’]:+.1f}%”
)
return “\n”.join(lines)

# =============================

# MOMENTUM SCANNER

# =============================

def scan_market(symbols: list, _counter=None, _pbar=None) -> pd.DataFrame:
results = []
failed  = []
total   = len(symbols)

```
if _pbar is None:
    _pbar = st.progress(0)
if _counter is None:
    _counter = st.empty()

for i in range(0, total, CHUNK_SIZE):
    tickers = symbols[i : i + CHUNK_SIZE]
    current = min(i + CHUNK_SIZE, total)
    _pbar.progress(int(current / total * 100))
    _counter.markdown(
        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.76rem;color:#8896b3'>"
        f"Equities &nbsp;<strong style='color:#d4a843'>{current}</strong> / {total}</span>",
        unsafe_allow_html=True,
    )
    try:
        data = yf.download(
            tickers, period="1y", interval="1d",
            group_by="ticker", threads=True, progress=False, auto_adjust=True,
        )
    except Exception as e:
        failed.extend([(t, str(e)) for t in tickers])
        continue

    for ticker in tickers:
        try:
            df = data[ticker] if len(tickers) > 1 else data
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) < MIN_BARS:
                continue
            price     = float(df["Close"].iloc[-1])
            ma50      = float(df["Close"].rolling(MA_PERIOD).mean().iloc[-1])
            high20    = float(df["High"].iloc[-(BREAKOUT_LOOKBACK + 1):-1].max())
            vol_avg   = float(df["Volume"].iloc[-(VOL_LOOKBACK + 1):-1].mean())
            vol_now   = float(df["Volume"].iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
            rsi       = compute_rsi(df["Close"])
            upside    = estimate_upside(df, price, high20)
            is_breakout  = price >= high20 * BREAKOUT_TOLERANCE
            is_vol_surge = vol_ratio > VOL_SURGE_THRESH
            if price > ma50 and (is_breakout or is_vol_surge):
                results.append({
                    "Ticker"       : ticker.replace(".NS", ""),
                    "Price ₹"      : round(price, 2),
                    "MA50"         : round(ma50, 2),
                    "RSI"          : rsi,
                    "Vol Ratio"    : round(vol_ratio, 2),
                    "Signal"       : "Breakout" if is_breakout else "Building",
                    "Target ₹"     : upside["target"],
                    "Upside %"     : upside["upside_pct"],
                    "52W High ₹"   : upside["high52"],
                    "Gap to 52W %" : upside["high52_gap_pct"],
                })
        except Exception as e:
            failed.append((ticker, str(e)))

if _pbar:
    _pbar.empty()
if _counter:
    _counter.empty()
if failed:
    st.caption(f"⚠ {len(failed)} tickers skipped. First 5: {[f[0] for f in failed[:5]]}")
if not results:
    return pd.DataFrame()
return (
    pd.DataFrame(results)
    .sort_values(["Signal", "Vol Ratio"], ascending=[True, False])
    .reset_index(drop=True)
)
```

# =============================

# LIVE MARKET SNAPSHOT — fetch current Nifty, Sensex, INR from yfinance

# =============================

def get_live_market_snapshot() -> str:
“”“Fetch real-time Nifty50, Bank Nifty, USD/INR, Brent crude levels.”””
symbols = {
“^NSEI”  : “Nifty 50”,
“^NSEBANK”: “Bank Nifty”,
“USDINR=X”: “USD/INR”,
“BZ=F”   : “Brent Crude (USD/bbl)”,
}
lines = []
for sym, label in symbols.items():
try:
t    = yf.Ticker(sym)
hist = t.history(period=“2d”, interval=“1d”)
if hist.empty:
continue
price  = float(hist[“Close”].iloc[-1])
prev   = float(hist[“Close”].iloc[-2]) if len(hist) >= 2 else price
chg    = round((price / prev - 1) * 100, 2)
lines.append(f”- {label}: {price:,.2f} ({chg:+.2f}% vs prev close)”)
except Exception:
continue
return “\n”.join(lines) if lines else “Live data unavailable.”

# =============================

# NEWS FETCH

# =============================

def get_raw_news() -> str:
today = datetime.now(IST).strftime(”%d %B %Y”)
query = (
f”India stock market Nifty Sensex {today} RBI interest rate FII DII flows “
f”Brent crude oil INR USD rupee NSE BSE “
f”Nifty Bank IT pharma auto metals earnings sector”
)
lines = []
try:
result = tavily.search(query=query, max_results=8)
for i, r in enumerate(result.get(“results”, []), 1):
title   = r.get(“title”, “”).strip()
content = r.get(“content”, “”).strip()
if not title:
continue
lines.append(f”{i}. {title}\n   {content[:300]}”)
except Exception as e:
lines.append(f”News fetch error: {e}”)
return “\n\n”.join(lines) if lines else “No news retrieved.”

# =============================

# AI MARKET INTELLIGENCE SUMMARY

# =============================

def generate_intel_summary(raw_news: str, live_snapshot: str) -> str:
system = (
“You are a senior Indian market analyst. Use the LIVE MARKET DATA as ground truth “
“for current index levels and prices — never contradict it. Supplement with news context. “
“Be specific with numbers. Do not reproduce headlines.”
)
prompt = f””“Today: {datetime.now(IST).strftime(’%d %B %Y, %H:%M IST’)}

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
“””
result = call_ai(prompt, system=system, max_tokens=6000)
return result if result.strip() else “Market intelligence could not be generated.”

# =============================

# SECTOR AGGREGATION

# =============================

def build_sector_heatmap(df: pd.DataFrame) -> pd.DataFrame:
if df.empty:
return pd.DataFrame()
rows = []
for _, r in df.iterrows():
sector = SECTOR_MAP.get(r[“Ticker”])
if sector is None:
continue
rows.append({“Sector”: sector, “Signal”: r[“Signal”],
“Ticker”: r[“Ticker”], “Upside %”: r[“Upside %”]})
if not rows:
return pd.DataFrame()
agg     = pd.DataFrame(rows)
summary = (
agg.groupby(“Sector”)
.agg(
Breakouts   =(“Signal”,   lambda x: (x == “Breakout”).sum()),
Building    =(“Signal”,   lambda x: (x == “Building”).sum()),
Avg_Upside  =(“Upside %”, “mean”),
Top_Tickers =(“Ticker”,   lambda x: “, “.join(x.tolist()[:4])),
)
.reset_index()
)
summary[“Total Signals”] = summary[“Breakouts”] + summary[“Building”]
summary[“Avg Upside %”]  = summary[“Avg_Upside”].round(1)
return (
summary[[“Sector”,“Total Signals”,“Breakouts”,“Building”,“Avg Upside %”,“Top_Tickers”]]
.sort_values(“Total Signals”, ascending=False)
.reset_index(drop=True)
)

def build_sector_context(sector_df: pd.DataFrame) -> str:
if sector_df.empty:
return “No named-sector data from scan.”
return “\n”.join(
f”- {r[‘Sector’]}: {r[‘Total Signals’]} signals “
f”({r[‘Breakouts’]} breakouts, {r[‘Building’]} building) | “
f”avg upside {r[‘Avg Upside %’]}%”
for _, r in sector_df.head(6).iterrows()
)

def build_momentum_context(df: pd.DataFrame) -> str:
if df.empty:
return “No momentum candidates identified.”
return “\n”.join(
f”{r[‘Ticker’]} ₹{r[‘Price ₹’]} {r[‘Signal’]} RSI{r[‘RSI’]} “
f”Vol{r[‘Vol Ratio’]}x Target₹{r[‘Target ₹’]}(+{r[‘Upside %’]}%)”
for _, r in df.head(20).iterrows()
)

def condense_intel(intel_summary: str) -> str:
“”“Extract only the key numbers/facts from intel summary for the strategy prompt.
Keeps the strategy prompt small while preserving actionable macro context.”””
key_lines = []
for line in intel_summary.splitlines():
line = line.strip()
if not line or line.startswith(”#”):
continue
if any(c in line for c in [“₹”, “$”, “%”, “•”, “*”, “-”, “·”]) or   
any(k in line.lower() for k in [“nifty”,“sensex”,“rbi”,“fii”,“dii”,“crude”,“brent”,“inr”,“usd”,“rate”,“inflation”,“gdp”]):
key_lines.append(line.lstrip(”•*- “))
return “\n”.join(key_lines[:12]) if key_lines else intel_summary[:400]

# =============================

# AI STRATEGY BRIEF

# =============================

def generate_strategy(
intel_summary: str,
momentum_context: str,
sector_context: str,
etf_context: str,
) -> str:
macro_facts = condense_intel(intel_summary)

```
etf_lines    = [l for l in etf_context.splitlines() if l.strip()][:8]
etf_short    = "\n".join(etf_lines)

stock_lines  = [l for l in momentum_context.splitlines() if l.strip()][:12]
stocks_short = "\n".join(stock_lines)

sector_lines = [l for l in sector_context.splitlines() if l.strip()][:4]
sector_short = "\n".join(sector_lines)

system = "NSE equity analyst. Cite both technical AND macro reason per pick. Only use tickers from STOCKS. TOP 3 ≠ AVOID."

prompt = f"""{datetime.now(IST).strftime('%d %b %Y')}
```

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
|---|----|-----|---|---|------|

**WATCHLIST** 3 stocks from STOCK SCAN, one bullet each.
**SECTOR TO OWN** 1 sector+ETF ticker, 2 sentences.
**AVOID** 2 bullets, not Top 3 picks.
**TAIL RISKS** 2 bullets.”””

```
result = call_ai(prompt, system=system, max_tokens=6000)
if result.strip():
    return result

# Retry with minimal prompt
mini = f"""{datetime.now(IST).strftime('%d %b %Y')}
```

MACRO:{macro_facts[:300]}
STOCK SCAN (picks must come ONLY from here, no ETFs):{stocks_short[:400]}
Write: **MARKET PULSE** 2 sentences. **TOP 3 BUY PICKS** table from STOCK SCAN only. **AVOID** 2 bullets.”””
result = call_ai(mini, system=system, max_tokens=6000)
return result if result.strip() else “Strategy unavailable — API timeout. Scan data above is valid.”

# =============================

# HELPERS

# =============================

def best_upside_picks(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
“””
Score by composite: upside% × vol_ratio × signal_weight × rsi_health
“””
if df.empty:
return pd.DataFrame()
d = df.copy()

```
d = d[d["RSI"] <= RSI_OVERBOUGHT] if "RSI" in d.columns else d
if d.empty:
    d = df.copy()

d["_sig_w"]  = d["Signal"].map({"Breakout": 1.4, "Building": 1.0}).fillna(1.0)

def rsi_score(r):
    if 40 <= r <= 65:  return 1.3
    if 30 <= r < 40:   return 1.0
    if 65 < r <= 72:   return 0.8
    return 0.6
d["_rsi_w"]  = d["RSI"].apply(rsi_score) if "RSI" in d.columns else 1.0
d["_vol_w"]  = d["Vol Ratio"].clip(upper=3.0) if "Vol Ratio" in d.columns else 1.0
d["_score"]  = d["Upside %"] * d["_sig_w"] * d["_rsi_w"] * d["_vol_w"]

return (
    d.nlargest(n, "_score")[["Ticker","Price ₹","Target ₹","Upside %","Signal","RSI"]]
    .reset_index(drop=True)
)
```

# =============================

# RUN BUTTON

# =============================

col_btn, col_note = st.columns([1, 3])
with col_btn:
run = st.button(“⚡  Run Market Intelligence”)
with col_note:
st.markdown(
“<small style='color:#4a5578;font-family:JetBrains Mono,monospace;font-size:0.7rem'>”
“~2,000 NSE EQ stocks  ·  ETF live data  ·  “
“AI news + strategy  ·  ~5–7 min”
“</small>”,
unsafe_allow_html=True,
)

if run:
if “last_run” in st.session_state:
elapsed = (datetime.now(IST) - st.session_state[“last_run”]).seconds
if elapsed < 120:
st.warning(f”⏳ Please wait {120 - elapsed}s before re-running.”)
st.stop()

```
st.session_state["last_run"] = datetime.now(IST)

# ── Phase 1: Data collection ──────────────────────────────────────
st.markdown(
    "<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
    "color:#4a5578;letter-spacing:0.12em;text-transform:uppercase'>"
    "Phase 1 &nbsp;·&nbsp; Market Data</span>",
    unsafe_allow_html=True,
)

_eq_status = st.empty()
_eq_status.markdown(
    "<span style='font-family:JetBrains Mono,monospace;font-size:0.76rem;color:#8896b3'>"
    "Loading equity list …</span>", unsafe_allow_html=True
)
registry = get_nse_registry()
if registry.empty:
    st.error("Could not load NSE registry.")
    st.stop()
_eq_status.empty()

etf_registry = get_etf_registry()
n_etf_total  = len(etf_registry) if not etf_registry.empty else 0
_etf_counter = st.empty()
_etf_pbar    = st.progress(0)
etf_df = scan_etfs(etf_registry, _etf_counter, _etf_pbar) if not etf_registry.empty else pd.DataFrame()
_etf_counter.empty()
_etf_pbar.empty()

eq_symbols   = registry["YF_SYMBOL"].tolist()
n_eq_total   = len(eq_symbols)
_eq_counter  = st.empty()
_eq_pbar     = st.progress(0)
momentum_df  = scan_market(eq_symbols, _eq_counter, _eq_pbar)
_eq_counter.empty()
_eq_pbar.empty()

n_etfs_clean = len(etf_df) if not etf_df.empty else 0
st.markdown(
    f"<div style='font-family:JetBrains Mono,monospace;font-size:0.74rem;"
    f"color:#8896b3;display:flex;gap:28px;padding:6px 0 10px'>"
    f"<span>⬡ <strong style='color:#d4a843'>{n_eq_total:,}</strong> equities</span>"
    f"<span>◈ <strong style='color:#00c896'>{n_etfs_clean}</strong> ETFs</span>"
    f"<span>▲ <strong style='color:#f0c96a'>{len(momentum_df)}</strong> momentum signals</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── Phase 2: AI Analysis ──────────────────────────────────────────
st.markdown(
    "<span style='font-family:JetBrains Mono,monospace;font-size:0.7rem;"
    "color:#4a5578;letter-spacing:0.12em;text-transform:uppercase'>"
    "Phase 2 &nbsp;·&nbsp; AI Analysis</span>",
    unsafe_allow_html=True,
)
_ai_status = st.empty()
_ai_pbar   = st.progress(0)

def _ai(msg: str, pct: int):
    _ai_status.markdown(
        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.76rem;color:#8896b3'>"
        f"{msg}</span>", unsafe_allow_html=True
    )
    _ai_pbar.progress(pct)

_ai("Fetching live market prices …", 10)
live_snapshot = get_live_market_snapshot()

_ai("Pulling latest news …", 25)
raw_news = get_raw_news()

_ai("Analysing market intelligence …", 40)
intel_summary = generate_intel_summary(raw_news, live_snapshot)

_ai("Building strategy brief …", 70)
momentum_context = build_momentum_context(momentum_df)
sector_df        = build_sector_heatmap(momentum_df)
sector_context   = build_sector_context(sector_df)
etf_context      = build_etf_context(etf_df)
strategy         = generate_strategy(
    intel_summary, momentum_context, sector_context, etf_context
)

_ai_status.empty()
_ai_pbar.empty()

st.session_state.update({
    "momentum_df"  : momentum_df,
    "sector_df"    : sector_df,
    "etf_df"       : etf_df,
    "intel_summary": intel_summary,
    "strategy"     : strategy,
})
```

# =============================

# DISPLAY RESULTS

# =============================

if “strategy” in st.session_state:
momentum_df   = st.session_state[“momentum_df”]
intel_summary = st.session_state.get(“intel_summary”, “”)
strategy      = st.session_state[“strategy”]
etf_df        = st.session_state.get(“etf_df”, pd.DataFrame())

```
# ── KPI Row ──
if not momentum_df.empty:
    k1, k2, k3, k4 = st.columns(4)
    breakouts = int((momentum_df["Signal"] == "Breakout").sum())
    building  = int((momentum_df["Signal"] == "Building").sum())
    k1.metric("Momentum Stocks",   f"{len(momentum_df)}")
    k2.metric("Breakouts",         f"{breakouts}", f"+{building} building")
    k3.metric("Avg Upside Target", f"{momentum_df['Upside %'].mean():.1f}%")
    k4.metric("Best Upside Found", f"{momentum_df['Upside %'].max():.1f}%")

# ── Scanner Top Picks ──
if not momentum_df.empty:
    st.markdown('<div class="section-header">Scanner · Highest Upside Potential</div>', unsafe_allow_html=True)
    st.caption("Top 3 by upside % with RSI not overbought. AI Strategy Brief below makes its own picks using both technicals and macro/news — they may differ.")
    top3 = best_upside_picks(momentum_df, n=3)
    cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        sig_cls = "sig-breakout" if row["Signal"] == "Breakout" else "sig-building"
        with cols[idx]:
            st.markdown(f"""
            <div class="pick-card">
                <div class="ticker-name">{row['Ticker']}</div>
                <div class="ticker-meta">₹{row['Price ₹']} &nbsp;·&nbsp; RSI {row['RSI']}</div>
                <span class="upside-pill">▲ {row['Upside %']}%</span>
                <div class="target-row">Target &nbsp;<span class="target-price">₹{row['Target ₹']}</span></div>
                <div class="signal-tag {sig_cls}">{row['Signal'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("")

# ── Strategy + Intelligence ──
col_l, col_r = st.columns([3, 2])

with col_l:
    st.markdown('<div class="section-header">AI Strategy Brief</div>', unsafe_allow_html=True)
    if strategy:
        st.markdown(f'<div class="strategy-wrap">\n\n{strategy}\n\n</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠ Strategy brief could not be generated. Check the AI Debug Log below. The market intelligence and scan results are still available.")

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

# ── Full Momentum Table ──
st.markdown('<div class="section-header">Full Momentum Candidates</div>', unsafe_allow_html=True)
if not momentum_df.empty:
    def color_signal(val):
        return "color:#00c896;font-weight:700" if val == "Breakout" else "color:#f59e0b;font-weight:600"
    def color_upside(val):
        if val >= 20: return "color:#00c896;font-weight:700"
        if val >= 10: return "color:#d4a843"
        return ""
    def color_rsi_stock(val):
        return "color:#f87171;font-weight:700" if val > RSI_OVERBOUGHT else ""
    st.dataframe(
        momentum_df.style
            .applymap(color_signal,     subset=["Signal"])
            .applymap(color_upside,     subset=["Upside %"])
            .applymap(color_rsi_stock,  subset=["RSI"])
            .format({
                "Price ₹":"₹{:.2f}","MA50":"₹{:.2f}","Target ₹":"₹{:.2f}",
                "52W High ₹":"₹{:.2f}","RSI":"{:.1f}","Vol Ratio":"{:.2f}×",
                "Upside %":"+{:.1f}%","Gap to 52W %":"{:.1f}%",
            }),
        use_container_width=True, height=420,
    )
    st.caption(f"RSI > {RSI_OVERBOUGHT} highlighted red. All {len(momentum_df)} stocks above MA50 with breakout or volume signal.")
else:
    st.info("No momentum candidates found in this scan.")

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
```

else:
st.markdown(”””
<div class="empty-state">
<div class="empty-glyph">◈</div>
<div class="empty-title">
Hit  <strong style="color:#d4a843">Run Market Intelligence</strong>  to begin
</div>
<div class="empty-sub">
~2,000 NSE EQ stocks  ·  ETF data  ·  live prices  · 
AI strategy (technicals + news)  ·  full momentum table
</div>
</div>
“””, unsafe_allow_html=True)