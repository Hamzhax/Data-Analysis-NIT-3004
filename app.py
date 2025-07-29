# ===========================================================
# app.py — Data Dashboard Backend
# Version: 4.2 (final)  •  All features A→Z + rock‑solid auth
# ===========================================================
#
# ✅ Fixes the “logs out when navigating pages / refresh” issue:
#    • One consistent cookie: SESSION_COOKIE_NAME="dd_session"
#    • SameSite="Lax" by default (works for file:// and http://localhost)
#      -> use SAME_SITE=None + Secure=1 only if you proxy over HTTPS.
#    • CORS with supports_credentials + reflected Origin
#    • after_request adds ACA-* headers to every response
#    • OPTIONS handlers return 200 for any /api/* path
#
# ✅ Everything else we built: uploads, fetch-url, smartsearch, clean,
#    preview_json (GET/POST), manual analyses (summary, corr, vc, pca, kmeans,
#    assoc_rules), auto_explore bundle, AI summaries (Gemini), markdown & pdf
#    reports, correlation CSV/PNG export, admin user mgmt, state cache.
#
# ❗ Env vars you already use (see .env):
# GOOGLE_API_KEY, GOOGLE_CSE_ID, GEMINI_MODEL, UPLOAD_FOLDER, USERS_FILE,
# DATASETS_META, ALLOWED_ORIGINS, MAX_UPLOAD_MB, APP_SECRET ...
#
# Run:  python app.py   (PORT=5050 FLASK_DEBUG=1 optional)
# ===========================================================

import os, io, re, json, datetime, threading, time, copy, gc, traceback
from functools import lru_cache
from datetime import timedelta

from flask import (
    Flask, request, jsonify, session, send_from_directory,
    send_file, Blueprint, redirect, url_for
)
from flask_cors import CORS
from dotenv import load_dotenv

import pandas as pd
import numpy as np

# Optional deps
try: import requests
except Exception: requests = None
import time, copy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Additional sklearn imports for advanced analytics
try:
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from sklearn.cluster import DBSCAN
    from sklearn.manifold import TSNE
    from sklearn.decomposition import FastICA
    SKLEARN_ADVANCED_AVAILABLE = True
except Exception:
    SKLEARN_ADVANCED_AVAILABLE = False
    print("[Warning] Some sklearn components not available - advanced analytics may be limited")

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

# Statsmodels for time series analysis
try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False
    print("[Warning] statsmodels not available - time series analysis will be disabled")

from werkzeug.security import generate_password_hash, check_password_hash

# Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# PDF
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, blue, darkblue
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# PNG correlation
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

load_dotenv()

VERSION                = "4.2"
APP_SECRET             = os.getenv("APP_SECRET", "change_me_dev_secret")

GOOGLE_API_KEY         = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID          = os.getenv("GOOGLE_CSE_ID")
MODEL_NAME             = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

UPLOAD_FOLDER          = os.getenv("UPLOAD_FOLDER", "uploads")
USERS_FILE             = os.getenv("USERS_FILE", "users.json")
DATASETS_META_FILE     = os.getenv("DATASETS_META", "datasets.json")

MAX_UPLOAD_MB          = int(os.getenv("MAX_UPLOAD_MB", "80"))
CORR_MAX_SIDE          = int(os.getenv("CORR_MAX_SIDE", "60"))
CORR_MAX_AREA          = int(os.getenv("CORR_MAX_AREA", "3600"))

ALLOWED_ORIGINS        = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500").split(",") if o.strip()]

# Session / cookie tight config
SESSION_COOKIE_NAME    = os.getenv("SESSION_COOKIE_NAME", "dd_session")
SESSION_COOKIE_SAMESITE= os.getenv("SESSION_COOKIE_SAMESITE", "Lax")  # "Lax" recommended for dev
SESSION_COOKIE_SECURE  = bool(int(os.getenv("SESSION_COOKIE_SECURE", "0")))  # set 1 only if HTTPS

PERMANENT_LIFETIME_SEC = int(os.getenv("PERMANENT_SESSION_LIFETIME", "86400"))  # 1 day default

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------- Flask app -------------
app = Flask(__name__, static_folder="static", static_url_path="")
app.secret_key = APP_SECRET
app.config.update(
    MAX_CONTENT_LENGTH           = MAX_UPLOAD_MB * 1024 * 1024,
    UPLOAD_FOLDER                = UPLOAD_FOLDER,
    SESSION_COOKIE_NAME          = SESSION_COOKIE_NAME,
    SESSION_COOKIE_HTTPONLY      = True,
    SESSION_COOKIE_SAMESITE      = SESSION_COOKIE_SAMESITE,
    SESSION_COOKIE_SECURE        = SESSION_COOKIE_SECURE,
    PERMANENT_SESSION_LIFETIME   = timedelta(seconds=PERMANENT_LIFETIME_SEC),
)
# --- simple in‑memory cache (resets on dyno restart) ---
_BUNDLE_CACHE = {}
_CACHE_TTL_SEC = 600  # 10 min

def _get_cached_bundle(filename):
    item = _BUNDLE_CACHE.get(filename)
    if not item:
        return None
    bundle, ts = item
    if time.time() - ts > _CACHE_TTL_SEC:
        _BUNDLE_CACHE.pop(filename, None)
        return None
    return bundle

def _cache_bundle(filename, bundle):
    _BUNDLE_CACHE[filename] = (bundle, time.time())

def _top_corr_pairs(corr_dict, limit=50):
    pairs = []
    cols = list(corr_dict.keys())
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            v = corr_dict[a].get(b)
            if isinstance(v, (int, float)):
                pairs.append((a, b, abs(v), v))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [{"a": a, "b": b, "abs": av, "r": r} for a, b, av, r in pairs[:limit]]

def slim_bundle(bundle):
    """Return a smaller version that’s fast to ship to the browser."""
    out = copy.deepcopy(bundle)

    # Correlation
    if "correlation_matrix" in out:
        out["top_correlations"] = _top_corr_pairs(out["correlation_matrix"], 50)
        # drop full matrix if size is an issue (front-end can request /api/correlation/export)
        # del out["correlation_matrix"]

    # Association rules
    if "assoc_rules" in out and len(out["assoc_rules"]) > 100:
        out["assoc_rules"] = out["assoc_rules"][:100]
    
    # PCA - keep all data, it's already limited to 300 points
    if "pca" in bundle:
        out["pca"] = bundle["pca"]
    
    # K-means - keep all data, it's already limited to 300 points  
    if "kmeans" in bundle:
        out["kmeans"] = bundle["kmeans"]

    return out

# Always set permanent (rolling) sessions
@app.before_request
def _permanent():
    session.permanent = True

# CORS (reflected origin, creds)
CORS(app, origins=ALLOWED_ORIGINS, supports_credentials=True)

# ---------- Gemini init ----------
GEMINI_MODEL = None
if GOOGLE_API_KEY and genai:
    try:
        print(f"[Gemini] Initializing with API key: {GOOGLE_API_KEY[:10]}...")
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel(MODEL_NAME)
        print(f"[Gemini] Successfully initialized model: {MODEL_NAME}")
    except Exception as e:
        print(f"[Gemini] init failed: {e}")
        print(f"[Gemini] API key present: {bool(GOOGLE_API_KEY)}")
        print(f"[Gemini] genai available: {bool(genai)}")
else:
    print(f"[Gemini] Not initializing - API key present: {bool(GOOGLE_API_KEY)}, genai available: {bool(genai)}")

# ---------- Helpers ----------
def ok(**payload):            return jsonify({"status": "ok", **payload})
def fail(msg, code=400):      return jsonify({"status": "error", "error": msg}), code

_users_lock = threading.Lock()
_meta_lock  = threading.Lock()

def load_json_file(path, fallback):
    if not os.path.exists(path): return fallback
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception: return fallback

def save_json_file(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(data, f, indent=2)
    os.replace(tmp, path)

def load_users(): return load_json_file(USERS_FILE, [])
def save_users(users):
    with _users_lock: save_json_file(USERS_FILE, users)

def sanitize_filename(name: str):
    name = re.sub(r"[^\w\-. ]+", "_", name)
    return name[:120]

def find_user(username):
    username = username.lower()
    for u in load_users():
        if u.get("username", "").lower() == username:
            return u
    return None

def require_login():
    if "user" not in session:
        return False, fail("Authentication required.", 401)
    return True, None

def require_admin():
    if "user" not in session or session.get("role") != "admin":
        return False, fail("Admin privilege required.", 403)
    return True, None

def load_meta(): return load_json_file(DATASETS_META_FILE, {})
def save_meta(meta: dict):
    with _meta_lock: save_json_file(DATASETS_META_FILE, meta)

CSV_ENCODINGS_TRY = ["utf-8", "utf-8-sig", "latin1"]

def read_csv_resilient(path, **kwargs):
    last_err = None
    for enc in CSV_ENCODINGS_TRY:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err

@lru_cache(maxsize=32)
def load_df(filename: str) -> pd.DataFrame:
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        raise FileNotFoundError("File missing")
    return read_csv_resilient(path)

def invalidate_cache():
    load_df.cache_clear()

def update_dataset_metadata(filename):
    meta = load_meta()
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path): return
    stat = os.stat(path)
    rows = cols = None
    try:
        full = read_csv_resilient(path)
        rows, cols = full.shape
    except Exception:
        pass
    meta[filename] = {
        "filename": filename,
        "size_bytes": stat.st_size,
        "uploaded_at": datetime.datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "rows": rows,
        "columns": cols
    }
    save_meta(meta)

def safe_describe(df: pd.DataFrame):
    desc = {}
    for col in df.columns:
        try:
            stats = df[col].describe(include="all")
            d = {}
            for k, v in stats.to_dict().items():
                if pd.isna(v):
                    d[k] = None
                else:
                    try:
                        d[k] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else (str(v)[:500])
                    except Exception:
                        d[k] = str(v)[:500]
            desc[col] = d
        except Exception:
            desc[col] = {"error": "describe_failed"}
    return desc

def compress_hist(series, bins=12):
    try:
        clean = series.dropna()
        if clean.empty: return []
        hist, edges = np.histogram(clean, bins=bins)
        return [{"start": float(edges[i]), "end": float(edges[i+1]), "count": int(hist[i])} for i in range(len(hist))]
    except Exception:
        return []

def top_cats(series, top=8):
    try:
        vc = series.astype(str).value_counts().head(top)
        return [{"value": idx, "count": int(cnt)} for idx, cnt in vc.items()]
    except Exception:
        return []

def infer_column_types(df: pd.DataFrame):
    types = {}
    for c in df.columns:
        s = df[c]
        kind = "TEXT"
        if pd.api.types.is_numeric_dtype(s):
            kind = "NUM"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "DATE"
        else:
            uni = s.nunique(dropna=True)
            if uni <= 2:
                kind = "BOOL"
            elif uni <= max(20, int(0.05 * len(s))):
                kind = "CAT"
            if re.search(r"(id|uuid|guid|code|ref)$", c, re.I):
                kind = "ID"
            if re.search(r"(date|time|timestamp|dt)$", c, re.I):
                kind = "DATE"
        types[c] = kind
    return types

def maybe_truncate_correlation(corr: pd.DataFrame):
    side = corr.shape[1]
    area = side * side
    truncated = False
    kept_cols = list(corr.columns)
    if side > CORR_MAX_SIDE or area > CORR_MAX_AREA:
        truncated = True
        abs_corr = corr.abs()
        importance = (abs_corr.sum(axis=0) - 1)
        top_cols = importance.sort_values(ascending=False).head(min(CORR_MAX_SIDE, side)).index.tolist()
        corr = corr.loc[top_cols, top_cols]
        kept_cols = top_cols
    return corr, truncated, kept_cols, side

# ---------- STATE CACHE (optional micro-store) ----------
state_bp = Blueprint("state", __name__, url_prefix="/api/state")
_state_cache = {}

@state_bp.post("/bundle")
def save_bundle():
    ok_login, resp = require_login()
    if not ok_login: return resp
    b = request.get_json(force=True) or {}
    file_id = b.get("file_id") or session.get("filename") or "unknown"
    sid = session.setdefault("sid", session.get("user", "anon"))
    key = f"{sid}:{file_id}:bundle"
    _state_cache[key] = b.get("bundle")
    return ok(key=key)

@state_bp.get("/bundle/<file_id>")
def get_bundle(file_id):
    ok_login, resp = require_login()
    if not ok_login: return resp
    sid = session.setdefault("sid", session.get("user", "anon"))
    key = f"{sid}:{file_id}:bundle"
    return ok(bundle=_state_cache.get(key))

app.register_blueprint(state_bp)

# ---------- AUTH ----------
@app.post("/api/register")
def register():
    try:
        data = request.get_json(force=True) or {}
        username = sanitize_filename((data.get("username") or "").strip())
        password = data.get("password") or ""
        role = "admin" if data.get("role") == "admin" else "user"
        if not username or not password:
            return fail("Username & password required.")
        if find_user(username):
            return fail("User exists.")
        users = load_users()
        users.append({
            "username": username,
            "password_hash": generate_password_hash(password),
            "role": role
        })
        save_users(users)
        return ok(message="Registered.")
    except Exception as e:
        return fail(str(e), 500)

@app.post("/api/login")
def login():
    try:
        d = request.get_json(force=True) or {}
        u = (d.get("username") or "").strip()
        p = d.get("password") or ""
        user = find_user(u)
        if not user or not check_password_hash(user["password_hash"], p):
            return fail("Invalid credentials.", 401)
        session.permanent = True
        session["user"]  = user["username"]
        session["role"]  = user["role"]
        session.setdefault("sid", user["username"])
        return ok(user=user["username"], role=user["role"])
    except Exception as e:
        return fail(str(e), 500)
@app.get("/healthz")
def healthz():
    return "ok", 200
import secrets

@app.post("/api/forgot")
def forgot():
    d = request.get_json(force=True) or {}
    username = (d.get("username") or "").strip()
    if not username:
        return fail("Username required.")
    users = load_users()
    for u in users:
        if u["username"].lower() == username.lower():
            # Generate a temp password
            temp_pw = secrets.token_urlsafe(8)
            u["password_hash"] = generate_password_hash(temp_pw)
            save_users(users)
            # In production, send email here. For demo, just print.
            print(f"[FORGOT PASSWORD] User: {username}, Temp Password: {temp_pw}")
            return ok(message="Temporary password set. Please check your email (demo: check server logs).")
    return fail("User not found.", 404)
@app.post("/api/logout")
def logout():
    session.clear()
    return ok(message="Logged out.")

@app.get("/api/me")
def me():
    if "user" not in session:
        return fail("Not logged in.", 401)
    return ok(user=session["user"], role=session.get("role"))

# ---------- ADMIN ----------
@app.get("/api/admin/users")
def admin_users():
    ok_admin, resp = require_admin()
    if not ok_admin: return resp
    users = load_users()
    slim = [{"username": u["username"], "role": u["role"]} for u in users]
    return ok(users=slim)

@app.post("/api/admin/users")
def admin_create_user():
    ok_admin, resp = require_admin()
    if not ok_admin: return resp
    d = request.get_json(force=True) or {}
    username = sanitize_filename((d.get("username") or "").strip())
    password = d.get("password") or ""
    role     = "admin" if d.get("role") == "admin" else "user"
    if not username or not password:
        return fail("Username/password required.")
    if find_user(username):
        return fail("User exists.")
    users = load_users()
    users.append({"username": username, "password_hash": generate_password_hash(password), "role": role})
    save_users(users)
    return ok(message="created")

@app.delete("/api/admin/users/<username>")
def admin_delete_user(username):
    ok_admin, resp = require_admin()
    if not ok_admin: return resp
    users = load_users()
    new = [u for u in users if u["username"].lower() != username.lower()]
    if len(new) == len(users):
        return fail("Not found", 404)
    save_users(new)
    return ok(message="deleted")

@app.put("/api/admin/users/<username>/role")
def admin_set_role(username):
    ok_admin, resp = require_admin()
    if not ok_admin: return resp
    d = request.get_json(force=True) or {}
    role = d.get("role")
    if role not in ("user", "admin"):
        return fail("Invalid role.")
    users = load_users()
    changed = False
    for u in users:
        if u["username"].lower() == username.lower():
            u["role"] = role
            changed = True
    if not changed:
        return fail("Not found", 404)
    save_users(users)
    return ok(message="role updated")

# ---------- SEARCH / FETCH / UPLOAD ----------
@app.post("/api/smartsearch")
def smartsearch():
    ok_login, resp = require_login()
    if not ok_login: return resp
    if not (GOOGLE_API_KEY and GOOGLE_CSE_ID and requests):
        return fail("Google API not configured.", 500)
    d = request.get_json(force=True) or {}
    q = (d.get("query") or "").strip()
    if not q: return fail("Query required.")
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": GOOGLE_API_KEY, "cx": GOOGLE_CSE_ID, "q": q, "fileType": "csv", "num": 10},
            timeout=30
        )
        js = r.json()
        links = []
        for item in js.get("items", []) or []:
            link = item.get("link", "")
            if ".csv" in link.lower():
                links.append(link)
        return ok(links=links)
    except Exception as e:
        return fail(f"Search error: {e}", 500)

@app.post("/api/fetch-url")
def fetch_url():
    ok_login, resp = require_login()
    if not ok_login: return resp
    if requests is None:
        return fail("Requests not available.")
    d = request.get_json(force=True) or {}
    url = (d.get("url") or "").strip()
    if not url or ".csv" not in url.lower():
        return fail("Invalid CSV URL.")
    filename = sanitize_filename(os.path.basename(url.split("?")[0]) or "remote.csv")
    if not filename.lower().endswith(".csv"):
        filename += ".csv"
    path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        return fail(f"Download failed: {e}", 500)
    session["filename"] = filename
    invalidate_cache()
    update_dataset_metadata(filename)
    return ok(filename=filename)

@app.post("/api/upload")
def upload():
    ok_login, resp = require_login()
    if not ok_login: return resp
    if "file" not in request.files:
        return fail("No file.")
    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return fail("Only CSV allowed.")
    filename = sanitize_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    session["filename"] = filename
    invalidate_cache()
    update_dataset_metadata(filename)
    return ok(filename=filename)

@app.get("/api/files")
def list_files():
    ok_login, resp = require_login()
    if not ok_login: return resp
    files = [f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".csv")]
    meta = load_meta()
    enriched = []
    for f in files:
        m = meta.get(f, {})
        m["filename"] = f
        enriched.append(m)
    return ok(files=enriched, active=session.get("filename"))

@app.delete("/api/admin/files/<filename>")
def admin_delete_file(filename):
    ok_admin, resp = require_admin()
    if not ok_admin: return resp
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path):
        return fail("File not found", 404)
    os.remove(path)
    meta = load_meta()
    if filename in meta:
        del meta[filename]
    save_meta(meta)
    if session.get("filename") == filename:
        session.pop("filename", None)
    invalidate_cache()
    return ok(message="deleted")

@app.post("/api/set_active")
def set_active():
    ok_login, resp = require_login()
    if not ok_login: return resp
    d = request.get_json(force=True) or {}
    fn = d.get("filename")
    if not fn: return fail("filename required.")
    path = os.path.join(UPLOAD_FOLDER, fn)
    if not os.path.exists(path):
        return fail("Not found.")
    session["filename"] = fn
    return ok(active=fn)

# ---------- PREVIEW ----------
@app.get("/api/preview/<filename>")
def preview(filename):
    ok_login, resp = require_login()
    if not ok_login: return resp
    try:
        return send_from_directory(UPLOAD_FOLDER, filename)
    except Exception as e:
        return fail(str(e), 500)

@app.route("/api/preview_json", methods=["GET", "POST"])
def preview_json():
    ok_login, resp = require_login()
    if not ok_login: return resp
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        fn = data.get("filename") or session.get("filename")
    else:
        fn = request.args.get("filename") or session.get("filename")
    if not fn:
        return fail("No active file.")
    df = load_df(fn)
    head = df.head(12)
    return ok(filename=fn, columns=head.columns.tolist(), rows=head.to_dict(orient="records"))

# ---------- CLEAN ----------
@app.post("/api/clean")
def clean():
    ok_login, resp = require_login()
    if not ok_login: return resp
    d = request.get_json(force=True) or {}
    filename = d.get("filename") or session.get("filename")
    if not filename: return fail("Filename required.")
    path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(path): return fail("Not found.")
    df = read_csv_resilient(path)
    orig_shape = df.shape
    changed = False
    if d.get("remove_duplicates"):
        df2 = df.drop_duplicates()
        if not df2.equals(df): changed = True
        df = df2
    if d.get("drop_na"):
        df2 = df.dropna()
        if not df2.equals(df): changed = True
        df = df2
    fill_value = d.get("fill_value")
    if fill_value not in [None, ""]:
        df2 = df.fillna(fill_value)
        if not df2.equals(df): changed = True
        df = df2
    if changed:
        df.to_csv(path, index=False)
        invalidate_cache()
        update_dataset_metadata(filename)
        if _BUNDLE_CACHE.pop(filename, None):
            pass
        return ok(message="cleaned")
    else:
        return ok(message="already_clean")
@app.get("/api/coltypes")
def coltypes():
    ok_login, resp = require_login()
    if not ok_login: return resp
    fn = session.get("filename")
    if not fn: return fail("No active dataset.")
    try:
        df = load_df(fn)
        return ok(types=infer_column_types(df))
    except Exception as e:
        return fail(str(e), 500)

# ---------- ANALYZE ----------
VALID_METHODS = {
    "summary", "correlation", "value_counts", "pca", "kmeans", "assoc_rules",
    "linear_regression", "logistic_regression", "random_forest", "time_series_decomp",
    "outlier_detection", "feature_importance", "trend_analysis", "clustering_analysis",
    "anomaly_detection", "dimensionality_reduction", "regression_comparison", "classification_comparison"
}

@app.post("/api/analyze")
def analyze():
    ok_login, resp = require_login()
    if not ok_login: return resp
    body = request.get_json(force=True) or {}
    method = body.get("method")
    if method not in VALID_METHODS:
        return fail("Invalid method.")
    filename = session.get("filename")
    if not filename: return fail("No active dataset.")
    
    # Load and optimize dataset size for memory constraints
    df_full = load_df(filename)
    MAX_ROWS = 10000
    if df_full.shape[0] > MAX_ROWS:
        df = df_full.sample(n=MAX_ROWS, random_state=42)
        print(f"[Analysis] Using sample of {MAX_ROWS} rows from {df_full.shape[0]} total rows")
    else:
        df = df_full
    
    column = body.get("column")
    k = int(body.get("k", 3))

    try:
        if method == "summary":
            return ok(method=method, summary=safe_describe(df))

        if method == "correlation":
            num = df.select_dtypes(include="number")
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            corr_full = num.corr().fillna(0)
            corr_small, truncated, kept, orig_side = maybe_truncate_correlation(corr_full)
            return ok(method=method,
                      correlation=corr_small.to_dict(),
                      columns=list(corr_small.columns),
                      truncated=truncated,
                      kept_columns=kept,
                      original_columns=list(corr_full.columns),
                      original_side=orig_side)

        if method == "value_counts":
            if not column or column not in df.columns: return fail("Column missing/invalid.")
            counts = df[column].astype(str).value_counts().head(50)
            return ok(method=method, labels=counts.index.tolist(), values=counts.tolist(), title=f"Value Counts {column}")

        if method == "pca":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            pca = PCA(n_components=min(3, num.shape[1]), random_state=42)
            comps = pca.fit_transform(num)
            return ok(method=method,
                      components=[[float(a), float(b)] for a, b in comps[:, :2].tolist()],
                      explained_variance=pca.explained_variance_ratio_.tolist(),
                      columns=num.columns.tolist())

        if method == "kmeans":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            if num.shape[0] < k: return fail("Rows less than k.")
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labs = km.fit_predict(num)
            
            # Create 2D components for visualization
            components_2d = None
            if num.shape[1] >= 2:
                from sklearn.decomposition import PCA
                viz_pca = PCA(n_components=2, random_state=42)
                components_2d = viz_pca.fit_transform(num).tolist()
            
            result = {
                "labels": labs.tolist(), 
                "centers": km.cluster_centers_.tolist(), 
                "columns": num.columns.tolist()
            }
            if components_2d:
                result["components_2d"] = components_2d
                
            return ok(method=method, **result)

        if method == "assoc_rules":
            if not MLXTEND_AVAILABLE: return fail("mlxtend missing.")
            cats = df.select_dtypes(exclude="number").fillna("MISSING")
            if cats.empty: return fail("No categorical columns.")
            subset = cats.iloc[:, :8]
            encoded = pd.get_dummies(subset)
            freq = apriori(encoded, min_support=0.05, use_colnames=True)
            if freq.empty: return fail("No frequent itemsets.")
            rules = association_rules(freq, metric="confidence", min_threshold=0.6)
            if rules.empty: return fail("No rules found.")
            top = rules.sort_values("lift", ascending=False).head(50)

            def fs(x): return list(x) if isinstance(x, frozenset) else x

            recs = [{
                "antecedents": fs(r["antecedents"]),
                "consequents": fs(r["consequents"]),
                "support": float(r["support"]),
                "confidence": float(r["confidence"]),
                "lift": float(r["lift"])
            } for _, r in top.iterrows()]
            return ok(method=method, rules=recs)

        # Predictive Analysis Methods
        if method == "linear_regression":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            target_col = body.get("target") or num.columns[-1]  # Use last column as default target
            if target_col not in num.columns: return fail("Target column not found.")
            
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error
            
            X = num.drop(columns=[target_col])
            y = num[target_col]
            
            if len(X.columns) == 0: return fail("No features available after removing target.")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            feature_importance = [{"feature": col, "coefficient": float(coef)} 
                                for col, coef in zip(X.columns, model.coef_)]
            
            return ok(method=method, target=target_col, r2_score=float(r2), 
                     mse=float(mse), feature_importance=feature_importance,
                     predictions=[[float(actual), float(pred)] for actual, pred in zip(y_test[:50], y_pred[:50])])

        if method == "outlier_detection":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outliers = iso_forest.fit_predict(num)
            
            outlier_indices = np.where(outliers == -1)[0]
            outlier_scores = iso_forest.score_samples(num)
            
            # Get the most extreme outliers
            extreme_indices = np.argsort(outlier_scores)[:20]
            
            outlier_data = []
            for idx in extreme_indices:
                outlier_data.append({
                    "index": int(idx),
                    "score": float(outlier_scores[idx]),
                    "values": {col: float(num.iloc[idx][col]) for col in num.columns[:5]}  # Limit columns
                })
            
            return ok(method=method, 
                     outlier_count=int(len(outlier_indices)),
                     total_points=int(len(num)),
                     outlier_percentage=float(len(outlier_indices) / len(num) * 100),
                     extreme_outliers=outlier_data)

        if method == "feature_importance":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 3: return fail("Need >=3 numeric columns.")
            target_col = body.get("target") or num.columns[-1]
            if target_col not in num.columns: return fail("Target column not found.")
            
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            
            X = num.drop(columns=[target_col])
            y = num[target_col]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            
            importance_data = [{"feature": col, "importance": float(imp)} 
                             for col, imp in zip(X.columns, rf.feature_importances_)]
            importance_data.sort(key=lambda x: x["importance"], reverse=True)
            
            return ok(method=method, target=target_col, 
                     feature_importance=importance_data,
                     score=float(rf.score(X_test, y_test)))

        if method == "trend_analysis":
            # Simple trend analysis for time series or sequential data
            num = df.select_dtypes(include="number")
            if num.empty: return fail("No numeric columns.")
            
            trends = {}
            for col in num.columns[:10]:  # Limit to first 10 columns
                values = num[col].dropna()
                if len(values) < 10: continue
                
                # Calculate trend using linear regression on index
                x = np.arange(len(values)).reshape(-1, 1)
                y = values.values
                
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(x, y)
                
                slope = float(lr.coef_[0])
                r2 = float(lr.score(x, y))
                
                trend_direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
                
                trends[col] = {
                    "slope": slope,
                    "r2": r2,
                    "direction": trend_direction,
                    "strength": "strong" if r2 > 0.7 else "moderate" if r2 > 0.3 else "weak"
                }
            
            return ok(method=method, trends=trends)

        if method == "random_forest":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            target_col = body.get("target") or num.columns[-1]
            if target_col not in num.columns: return fail("Target column not found.")
            
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import r2_score, mean_squared_error
            
            X = num.drop(columns=[target_col])
            y = num[target_col]
            
            if len(X.columns) == 0: return fail("No features available after removing target.")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            feature_importance = [{"feature": col, "importance": float(imp)} 
                                for col, imp in zip(X.columns, model.feature_importances_)]
            feature_importance.sort(key=lambda x: x["importance"], reverse=True)
            
            return ok(method=method, target=target_col, r2_score=float(r2), 
                     mse=float(mse), feature_importance=feature_importance[:10],
                     predictions=[[float(actual), float(pred)] for actual, pred in zip(y_test[:50], y_pred[:50])])

        if method == "logistic_regression":
            # Check if we have a binary target column
            target_col = body.get("target")
            if not target_col: return fail("Target column required for logistic regression.")
            if target_col not in df.columns: return fail("Target column not found.")
            
            # Check if target is binary/categorical
            unique_vals = df[target_col].nunique()
            if unique_vals > 10: return fail("Target has too many unique values for logistic regression.")
            
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare features (numeric only for simplicity)
            num = df.select_dtypes(include="number").dropna()
            if target_col in num.columns:
                X = num.drop(columns=[target_col])
            else:
                X = num
            
            if len(X.columns) == 0: return fail("No numeric features available.")
            
            # Prepare target
            y = df[target_col].dropna()
            # Align X and y indices
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            # Encode target if it's categorical
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            feature_importance = [{"feature": col, "coefficient": float(coef)} 
                                for col, coef in zip(X.columns, model.coef_[0])]
            
            return ok(method=method, target=target_col, accuracy=float(accuracy),
                     feature_importance=feature_importance,
                     class_labels=le.classes_.tolist(),
                     predictions=[[int(actual), int(pred)] for actual, pred in zip(y_test[:50], y_pred[:50])])

        if method == "feature_importance":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.feature_selection import mutual_info_regression
            
            target_col = body.get("target") or num.columns[-1]
            if target_col not in num.columns: return fail("Target column not found.")
            
            X = num.drop(columns=[target_col])
            y = num[target_col]
            
            if len(X.columns) == 0: return fail("No features available.")
            
            # Random Forest feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            rf_importance = list(zip(X.columns, rf.feature_importances_))
            
            # Mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_importance = list(zip(X.columns, mi_scores))
            
            # Correlation with target
            corr_importance = [(col, abs(X[col].corr(y))) for col in X.columns]
            
            return ok(method=method, target=target_col,
                     random_forest_importance=sorted(rf_importance, key=lambda x: x[1], reverse=True),
                     mutual_info_importance=sorted(mi_importance, key=lambda x: x[1], reverse=True),
                     correlation_importance=sorted(corr_importance, key=lambda x: x[1], reverse=True))

        if method == "time_series_decomp":
            # Check if we have a datetime column and a numeric target
            date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns
            if len(date_cols) == 0:
                # Try to convert string columns to datetime
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        df[col] = pd.to_datetime(df[col])
                        date_cols = [col]
                        break
                    except:
                        continue
            
            if len(date_cols) == 0: return fail("No datetime column found for time series analysis.")
            
            target_col = body.get("target")
            if not target_col: return fail("Target column required for time series decomposition.")
            if target_col not in df.select_dtypes(include="number").columns:
                return fail("Target must be numeric for time series analysis.")
            
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            date_col = date_cols[0]
            ts_data = df[[date_col, target_col]].dropna().sort_values(date_col)
            
            if len(ts_data) < 24: return fail("Need at least 24 data points for time series decomposition.")
            
            # Set date as index
            ts_data.set_index(date_col, inplace=True)
            
            # Perform decomposition
            decomposition = seasonal_decompose(ts_data[target_col], model='additive', period=min(12, len(ts_data)//2))
            
            # Limit data returned to avoid memory issues
            max_points = 1000
            step = max(1, len(ts_data) // max_points)
            
            return ok(method=method, 
                     target=target_col,
                     date_column=date_col,
                     trend=[float(x) if not pd.isna(x) else None for x in decomposition.trend[::step]],
                     seasonal=[float(x) if not pd.isna(x) else None for x in decomposition.seasonal[::step]],
                     residual=[float(x) if not pd.isna(x) else None for x in decomposition.resid[::step]],
                     dates=[str(d) for d in ts_data.index[::step]])

        if method == "clustering_analysis":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            if num.shape[0] < 6: return fail("Need at least 6 data points.")
            
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(num)
            
            results = {}
            
            # K-means with different k values
            kmeans_results = []
            for k in range(2, min(8, len(num)//2)):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
                labels = kmeans.fit_predict(X_scaled)
                silhouette = silhouette_score(X_scaled, labels)
                kmeans_results.append({"k": k, "silhouette": float(silhouette), "inertia": float(kmeans.inertia_)})
            results["kmeans"] = kmeans_results
            
            # DBSCAN
            try:
                dbscan = DBSCAN(eps=0.5, min_samples=3)
                dbscan_labels = dbscan.fit_predict(X_scaled)
                n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
                if n_clusters > 1:
                    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
                    results["dbscan"] = {"clusters": n_clusters, "silhouette": float(dbscan_silhouette)}
            except:
                results["dbscan"] = {"error": "DBSCAN failed"}
            
            # Hierarchical clustering
            try:
                hierarchical = AgglomerativeClustering(n_clusters=3)
                hier_labels = hierarchical.fit_predict(X_scaled)
                hier_silhouette = silhouette_score(X_scaled, hier_labels)
                results["hierarchical"] = {"clusters": 3, "silhouette": float(hier_silhouette)}
            except:
                results["hierarchical"] = {"error": "Hierarchical clustering failed"}
            
            return ok(method=method, results=results, columns=num.columns.tolist())

        if method == "anomaly_detection":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 1: return fail("Need at least 1 numeric column.")
            
            from sklearn.ensemble import IsolationForest
            from sklearn.svm import OneClassSVM
            from sklearn.covariance import EllipticEnvelope
            
            results = {}
            
            # Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_outliers = iso_forest.fit_predict(num)
            iso_scores = iso_forest.score_samples(num)
            results["isolation_forest"] = {
                "outliers": int(sum(iso_outliers == -1)),
                "outlier_percentage": float(sum(iso_outliers == -1) / len(num) * 100),
                "anomaly_scores": [float(score) for score in iso_scores[:100]]  # Limit to 100 for performance
            }
            
            # One-Class SVM
            try:
                svm = OneClassSVM(gamma='auto')
                svm_outliers = svm.fit_predict(num)
                results["one_class_svm"] = {
                    "outliers": int(sum(svm_outliers == -1)),
                    "outlier_percentage": float(sum(svm_outliers == -1) / len(num) * 100)
                }
            except:
                results["one_class_svm"] = {"error": "SVM anomaly detection failed"}
            
            # Elliptic Envelope
            try:
                envelope = EllipticEnvelope(contamination=0.1, random_state=42)
                env_outliers = envelope.fit_predict(num)
                results["elliptic_envelope"] = {
                    "outliers": int(sum(env_outliers == -1)),
                    "outlier_percentage": float(sum(env_outliers == -1) / len(num) * 100)
                }
            except:
                results["elliptic_envelope"] = {"error": "Elliptic Envelope detection failed"}
            
            return ok(method=method, results=results, total_points=len(num))

        if method == "dimensionality_reduction":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 3: return fail("Need >=3 numeric columns for dimensionality reduction.")
            
            from sklearn.decomposition import PCA, FastICA
            from sklearn.manifold import TSNE
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(num)
            
            results = {}
            
            # PCA
            pca = PCA(n_components=min(3, num.shape[1]), random_state=42)
            pca_result = pca.fit_transform(X_scaled)
            results["pca"] = {
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "components_2d": pca_result[:, :2].tolist(),
                "cumulative_variance": float(sum(pca.explained_variance_ratio_))
            }
            
            # t-SNE (only if reasonable number of samples)
            if len(num) <= 1000:
                try:
                    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(num)-1))
                    tsne_result = tsne.fit_transform(X_scaled)
                    results["tsne"] = {
                        "components_2d": tsne_result.tolist()
                    }
                except:
                    results["tsne"] = {"error": "t-SNE failed"}
            else:
                results["tsne"] = {"error": "Too many samples for t-SNE"}
            
            # ICA
            try:
                ica = FastICA(n_components=min(3, num.shape[1]), random_state=42)
                ica_result = ica.fit_transform(X_scaled)
                results["ica"] = {
                    "components_2d": ica_result[:, :2].tolist()
                }
            except:
                results["ica"] = {"error": "ICA failed"}
            
            return ok(method=method, results=results, original_features=num.columns.tolist())

        if method == "regression_comparison":
            num = df.select_dtypes(include="number").dropna()
            if num.shape[1] < 2: return fail("Need >=2 numeric columns.")
            target_col = body.get("target") or num.columns[-1]
            if target_col not in num.columns: return fail("Target column not found.")
            
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import r2_score, mean_squared_error
            
            X = num.drop(columns=[target_col])
            y = num[target_col]
            
            if len(X.columns) == 0: return fail("No features available.")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                "Lasso Regression": Lasso(alpha=1.0, random_state=42),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            results = []
            for name, model in models.items():
                try:
                    # Fit and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    
                    results.append({
                        "model": name,
                        "r2_score": float(r2),
                        "mse": float(mse),
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std())
                    })
                except Exception as e:
                    results.append({
                        "model": name,
                        "error": str(e)
                    })
            
            return ok(method=method, target=target_col, model_comparison=results)

        if method == "classification_comparison":
            target_col = body.get("target")
            if not target_col: return fail("Target column required for classification comparison.")
            if target_col not in df.columns: return fail("Target column not found.")
            
            # Check if target is suitable for classification
            unique_vals = df[target_col].nunique()
            if unique_vals > 20: return fail("Target has too many unique values for classification.")
            
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            from sklearn.naive_bayes import GaussianNB
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare features (numeric only)
            num = df.select_dtypes(include="number").dropna()
            if target_col in num.columns:
                X = num.drop(columns=[target_col])
            else:
                X = num
            
            if len(X.columns) == 0: return fail("No numeric features available.")
            
            # Prepare target
            y = df[target_col].dropna()
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
            
            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
            
            models = {
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Naive Bayes": GaussianNB(),
                "SVM": SVC(random_state=42, gamma='auto')
            }
            
            results = []
            for name, model in models.items():
                try:
                    # Fit and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
                    
                    results.append({
                        "model": name,
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "cv_mean": float(cv_scores.mean()),
                        "cv_std": float(cv_scores.std())
                    })
                except Exception as e:
                    results.append({
                        "model": name,
                        "error": str(e)
                    })
            
            return ok(method=method, target=target_col, 
                     model_comparison=results, class_labels=le.classes_.tolist())

    except Exception as e:
        return fail(f"Analysis failed: {e}", 500)

# ---------- AI SUMMARY ----------
@app.post("/api/ai_summary")
def ai_summary():
    ok_login, resp = require_login()
    if not ok_login: return resp
    if not GEMINI_MODEL:
        return fail("AI not configured.", 500)
    d = request.get_json(force=True) or {}
    ctype = (d.get("chart_type") or "").strip()
    desc  = (d.get("description") or "").strip()
    if not ctype or not desc:
        return fail("chart_type and description required.")
    prompt = f"""
You are a senior data analyst. Analyze dataset context: {ctype}
User focus: "{desc}"
Return STRICT JSON with keys:
summary (2 short sentences),
key_points (3-5 bullet strings),
anomalies (array, may be empty),
recommendation (single sentence).
If information insufficient, still produce generic safe suggestions.
"""
    try:
        resp_ai = GEMINI_MODEL.generate_content(prompt)
        raw = (getattr(resp_ai, "text", None) or "").strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"summary": raw[:600], "key_points": [], "anomalies": [], "recommendation": ""}
        return ok(**parsed)
    except Exception as e:
        return fail(f"AI summary failed: {e}", 500)

@app.post("/api/ai_chart_description")
def ai_chart_description():
    ok_login, resp = require_login()
    if not ok_login: return resp
    if not GEMINI_MODEL:
        return fail("AI not configured.", 500)
    
    d = request.get_json(force=True) or {}
    chart_type = (d.get("chart_type") or "").strip()
    data_context = d.get("data_context", {})
    
    if not chart_type:
        return fail("chart_type required.")
    
    # Create context-aware descriptions
    chart_descriptions = {
        "correlation_heatmap": {
            "purpose": "Shows the strength and direction of linear relationships between numeric variables",
            "interpretation": "Values range from -1 to 1. Values close to 1 or -1 indicate strong relationships, while values near 0 indicate weak relationships. Red colors typically show positive correlations, blue shows negative correlations.",
            "insights": "Use this to identify which variables move together, find redundant features, or discover unexpected relationships in your data."
        },
        "pca_scatter": {
            "purpose": "Visualizes high-dimensional data in 2D space while preserving as much variance as possible",
            "interpretation": "Each point represents a data record projected onto the first two principal components. Clustering patterns may reveal natural groupings in your data.",
            "insights": "Helpful for dimensionality reduction, outlier detection, and understanding the main patterns of variation in your dataset."
        },
        "kmeans_scatter": {
            "purpose": "Shows how K-means clustering has grouped your data points",
            "interpretation": "Different colors represent different clusters. Points close together are more similar. Cluster centers are typically marked distinctly.",
            "insights": "Use this to understand natural groupings in your data, customer segments, or to identify distinct patterns."
        },
        "histogram": {
            "purpose": "Shows the distribution of values for a single numeric variable",
            "interpretation": "The height of each bar represents frequency or count. The shape reveals if data is normal, skewed, has multiple peaks, or contains outliers.",
            "insights": "Essential for understanding data distribution, identifying outliers, and determining appropriate statistical methods."
        },
        "bar_chart": {
            "purpose": "Compares frequencies or values across different categories",
            "interpretation": "Bar height represents the count or measure for each category. Helps identify the most/least frequent categories.",
            "insights": "Perfect for categorical data analysis, identifying dominant categories, and comparing group sizes."
        },
        "scatter_plot": {
            "purpose": "Reveals relationships between two continuous variables",
            "interpretation": "Each point represents one observation. Patterns like lines suggest correlations, clusters suggest groups, and scattered points suggest no relationship.",
            "insights": "Use to identify correlations, trends, outliers, and to visualize the strength of relationships between variables."
        },
        "line_chart": {
            "purpose": "Shows trends and changes over time or ordered sequences",
            "interpretation": "The line connects data points in order, revealing trends, cycles, and patterns over time. Slopes indicate rate of change.",
            "insights": "Essential for time series analysis, trend identification, and understanding how variables change over time."
        },
        "box_plot": {
            "purpose": "Displays the distribution summary including median, quartiles, and outliers",
            "interpretation": "The box shows the middle 50% of data, whiskers show the range, and dots represent outliers. Useful for comparing distributions.",
            "insights": "Great for identifying outliers, comparing groups, and understanding data spread and central tendency."
        }
    }
    
    default_description = {
        "purpose": "Provides visual representation of your data patterns",
        "interpretation": "Examine the chart for trends, patterns, outliers, and relationships that may not be obvious in raw data",
        "insights": "Visual analysis often reveals insights that statistical summaries might miss"
    }
    
    description = chart_descriptions.get(chart_type, default_description)
    
    # Add data-specific context if available
    if data_context:
        context_prompt = f"""
Based on this chart type ({chart_type}) and data context: {data_context}, 
provide a specific interpretation focusing on what patterns or insights someone should look for.
Keep response concise (2-3 sentences) and actionable.
"""
        try:
            resp_ai = GEMINI_MODEL.generate_content(context_prompt)
            ai_context = (getattr(resp_ai, "text", None) or "").strip()
            if ai_context:
                description["ai_insights"] = ai_context
        except:
            pass  # If AI fails, just use the static description
    
    return ok(chart_type=chart_type, description=description)

# ---------- AUTO EXPLORE ----------
def generate_chart_descriptions(bundle):
    """Generate AI descriptions for all charts in the bundle"""
    descriptions = {}
    
    if not GEMINI_MODEL:
        # Return basic descriptions if AI not available
        return {
            "correlation_heatmap": {"description": "Shows correlation patterns between numeric variables"},
            "pca_scatter": {"description": "Visualizes dimensionality reduction in 2D space"},
            "cluster_scatter": {"description": "Shows clustering results and natural groupings"},
            "histogram": {"description": "Displays distribution of numeric values"},
            "bar_chart": {"description": "Compares frequencies across categories"},
            "time_series": {"description": "Shows trends and patterns over time"},
            "anomaly_detection": {"description": "Identifies outliers and unusual patterns"},
            "clustering_analysis": {"description": "Advanced clustering with multiple algorithms"},
            "dimensionality_reduction": {"description": "Multiple dimensionality reduction techniques"},
            "feature_importance": {"description": "Ranks features by their predictive importance"}
        }
    
    # Generate context-aware descriptions
    basic_info = bundle.get("profile", {}).get("basic", {})
    context = f"Dataset with {basic_info.get('rows', '?')} rows and {basic_info.get('columns', '?')} columns"
    
    chart_types = {
        "correlation_heatmap": bundle.get("correlation_matrix"),
        "pca_scatter": bundle.get("pca"),
        "cluster_scatter": bundle.get("kmeans"),
        "time_series": bundle.get("advanced_analytics", {}).get("time_series"),
        "anomaly_detection": bundle.get("advanced_analytics", {}).get("anomaly_detection"),
        "clustering_analysis": bundle.get("advanced_analytics", {}).get("clustering_analysis"),
        "dimensionality_reduction": bundle.get("advanced_analytics", {}).get("dimensionality_reduction"),
        "feature_importance": bundle.get("advanced_analytics", {}).get("feature_importance")
    }
    
    for chart_type, data in chart_types.items():
        if data:  # Only generate descriptions for available charts
            try:
                desc = ai_chart_description(chart_type, context)
                descriptions[chart_type] = desc
            except Exception as e:
                print(f"[Chart Description] Failed for {chart_type}: {e}")
                descriptions[chart_type] = {"description": f"Analysis results for {chart_type}"}
    
    return descriptions
    for chart_type, data in chart_types.items():
        if data:  # Only generate descriptions for available charts
            try:
                desc = ai_chart_description(chart_type, context)
                descriptions[chart_type] = desc
            except Exception as e:
                print(f"[Chart Description] Failed for {chart_type}: {e}")
                descriptions[chart_type] = {"description": f"Analysis results for {chart_type}"}
    
    return descriptions

def build_auto_bundle(filename):
    # Load the full DataFrame for preview/meta, but sample for heavy analysis
    df_full = load_df(filename)
    MAX_ROWS = 10000

    # Use a sample for all heavy analysis
    if df_full.shape[0] > MAX_ROWS:
        df = df_full.sample(n=MAX_ROWS, random_state=42)
    else:
        df = df_full

    num_df = df.select_dtypes(include="number")
    cat_df = df.select_dtypes(exclude="number")

    # Summary: use the sample (fast, but still representative)
    summary = safe_describe(df)

    # Categorical info
    categorical_info = {c: top_cats(cat_df[c], top=8) for c in cat_df.columns}
    # Numeric info
    numeric_info = {}
    for c in num_df.columns:
        numeric_info[c] = {
            "min":  float(num_df[c].min())  if not num_df[c].empty else None,
            "max":  float(num_df[c].max())  if not num_df[c].empty else None,
            "mean": float(num_df[c].mean()) if not num_df[c].empty else None,
            "std":  float(num_df[c].std())  if not num_df[c].empty else None,
            "hist": compress_hist(num_df[c])
        }

    # Correlation
    top_correlations = []
    correlation_matrix = None
    truncated = False
    kept_cols = []
    original_side = None
    if num_df.shape[1] >= 2:
        corr_full = num_df.corr().fillna(0)
        corr_small, truncated, kept_cols, original_side = maybe_truncate_correlation(corr_full)
        correlation_matrix = corr_small.to_dict()
        pairs = []
        cols = corr_full.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                pairs.append((cols[i], cols[j], float(corr_full.iloc[i, j])))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        top_correlations = pairs[:15]

    # PCA
    pca_result = None
    if num_df.shape[1] >= 2 and num_df.dropna().shape[0] > 5:
        nd = num_df.dropna()
        ncomp = min(3, nd.shape[1])
        try:
            pca = PCA(n_components=ncomp, random_state=42)
            comps = pca.fit_transform(nd)
            pca_result = {
                "explained_variance": pca.explained_variance_ratio_.tolist(),
                "components_2d": [[float(a), float(b)] for a, b in comps[:300, :2]]
            }
            print(f"[PCA] Generated PCA with {len(pca_result['components_2d'])} components")
            print(f"[PCA] Explained variance: {pca_result['explained_variance']}")
        except Exception as e:
            print(f"[PCA] Failed to generate PCA: {e}")
            pca_result = None
    else:
        print(f"[PCA] Insufficient data for PCA: {num_df.shape[1]} cols, {num_df.dropna().shape[0]} rows")

    # KMeans
    kmeans_result = None
    if num_df.shape[1] >= 2 and num_df.dropna().shape[0] >= 30:
        nd = num_df.dropna()
        max_k = min(6, max(3, nd.shape[0] // 8))
        inertias, models = [], []
        try:
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                km.fit(nd)
                inertias.append(km.inertia_)
                models.append(km)
            drops = []
            for i in range(1, len(inertias)):
                prev = inertias[i - 1]; cur = inertias[i]
                drop = (prev - cur) / prev if prev else 0
                drops.append((i + 2, drop))
            if drops:
                best_k = max(drops, key=lambda x: x[1])[0]
                best_model = models[best_k - 2]
                
                # Get 2D components for visualization - use PCA result if available
                components_2d = None
                if pca_result and "components_2d" in pca_result:
                    # Use the same subset of data that PCA used, but match kmeans length
                    pca_components = pca_result["components_2d"]
                    # Ensure we match the length of K-means labels
                    min_len = min(len(pca_components), len(best_model.labels_))
                    components_2d = pca_components[:min_len]
                elif nd.shape[1] >= 2:
                    # Create 2D projection for visualization if no PCA
                    from sklearn.decomposition import PCA
                    viz_pca = PCA(n_components=2, random_state=42)
                    components_2d = viz_pca.fit_transform(nd).tolist()
                
                kmeans_result = {
                    "k": best_k,
                    "centers": best_model.cluster_centers_.tolist(),
                    "labels_preview": best_model.labels_[:300].tolist(),
                    "labels": best_model.labels_.tolist(),
                    "columns": nd.columns.tolist()
                }
                
                # Add 2D components if available and ensure length consistency
                if components_2d:
                    # Make sure components_2d matches the length of labels_preview
                    max_len = min(len(components_2d), 300)
                    kmeans_result["components_2d"] = components_2d[:max_len]
                    # Also trim labels_preview to match
                    kmeans_result["labels_preview"] = best_model.labels_[:max_len].tolist()
                    
        except Exception as e:
            print(f"[KMeans] Failed to generate K-means: {e}")
            kmeans_result = None
    else:
        print(f"[KMeans] Insufficient data for K-means: {num_df.shape[1]} cols, {num_df.dropna().shape[0]} rows")

    # Association Rules
    assoc_result = None
    if MLXTEND_AVAILABLE:
        cats = cat_df.fillna("MISSING")
        if not cats.empty:
            sub = cats.iloc[:, :8]
            try:
                enc  = pd.get_dummies(sub)
                freq = apriori(enc, min_support=0.05, use_colnames=True)
                if not freq.empty:
                    rules = association_rules(freq, metric="confidence", min_threshold=0.6)
                    if not rules.empty:
                        top = rules.sort_values("lift", ascending=False).head(15)
                        def fs(x): return list(x) if isinstance(x, frozenset) else x
                        assoc_result = []
                        for _, r in top.iterrows():
                            assoc_result.append({
                                "antecedents": fs(r["antecedents"]),
                                "consequents": fs(r["consequents"]),
                                "support": float(r["support"]),
                                "confidence": float(r["confidence"]),
                                "lift": float(r["lift"])
                            })
            except Exception:
                assoc_result = None

    # Advanced Predictive Analytics (New)
    advanced_analytics = {}
    
    # Time Series Analysis (if data has datetime or index pattern)
    time_series_result = None
    if STATSMODELS_AVAILABLE:
        try:
            # Check for datetime columns or time-like patterns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0 and len(num_df.columns) > 0:
                dt_col = datetime_cols[0]
                target_col = num_df.columns[0]  # Use first numeric column
                
                # Create time series data
                ts_data = df[[dt_col, target_col]].dropna().set_index(dt_col).sort_index()
                
                if len(ts_data) >= 24:  # Minimum for seasonal decomposition
                    decomposition = seasonal_decompose(ts_data[target_col], model='additive', period=min(12, len(ts_data)//2))
                    time_series_result = {
                        "target_column": target_col,
                        "datetime_column": dt_col,
                        "trend": decomposition.trend.dropna().tolist()[:100],
                        "seasonal": decomposition.seasonal.dropna().tolist()[:100],
                        "residual": decomposition.resid.dropna().tolist()[:100],
                        "original": ts_data[target_col].tolist()[:100]
                    }
                    print(f"[Time Series] Generated analysis for {target_col}")
        except Exception as e:
            print(f"[Time Series] Failed: {e}")
            time_series_result = None

    # Anomaly Detection
    anomaly_result = None
    if num_df.shape[1] >= 1 and num_df.dropna().shape[0] >= 50:
        try:
            from sklearn.ensemble import IsolationForest
            
            num_clean = num_df.dropna()
            if len(num_clean) > 0:
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(num_clean)
                scores = iso_forest.score_samples(num_clean)
                
                anomaly_result = {
                    "method": "isolation_forest",
                    "outliers_count": int(sum(outliers == -1)),
                    "outlier_percentage": float(sum(outliers == -1) / len(outliers) * 100),
                    "anomaly_scores": [float(s) for s in scores[:100]],
                    "outlier_indices": [int(i) for i, x in enumerate(outliers) if x == -1][:50]
                }
                print(f"[Anomaly] Detected {anomaly_result['outliers_count']} outliers")
        except Exception as e:
            print(f"[Anomaly] Failed: {e}")
            anomaly_result = None

    # Clustering Analysis (Enhanced)
    clustering_analysis = None
    if num_df.shape[1] >= 2 and num_df.dropna().shape[0] >= 30:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler
            
            num_clean = num_df.dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(num_clean)
            
            # DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X_scaled)
            
            clustering_analysis = {
                "dbscan": {
                    "n_clusters": len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                    "n_noise": int(sum(dbscan_labels == -1)),
                    "labels": dbscan_labels[:300].tolist(),
                    "silhouette_samples": []
                }
            }
            
            # Add 2D projection for visualization
            if num_clean.shape[1] >= 2:
                from sklearn.decomposition import PCA
                pca_viz = PCA(n_components=2, random_state=42)
                components_2d = pca_viz.fit_transform(X_scaled)
                clustering_analysis["components_2d"] = components_2d[:300].tolist()
                
            print(f"[Clustering] DBSCAN found {clustering_analysis['dbscan']['n_clusters']} clusters")
        except Exception as e:
            print(f"[Clustering] Failed: {e}")
            clustering_analysis = None

    # Dimensionality Reduction (Enhanced)
    dim_reduction = None
    if num_df.shape[1] >= 3 and num_df.dropna().shape[0] >= 50:
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import FastICA
            from sklearn.preprocessing import StandardScaler
            
            num_clean = num_df.dropna()
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(num_clean)
            
            results = {}
            
            # t-SNE (if reasonable size)
            if len(num_clean) <= 1000:
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(num_clean)-1))
                tsne_result = tsne.fit_transform(X_scaled)
                results["tsne"] = tsne_result[:300].tolist()
            
            # ICA
            if num_clean.shape[1] >= 2:
                ica = FastICA(n_components=min(3, num_clean.shape[1]), random_state=42)
                ica_result = ica.fit_transform(X_scaled)
                results["ica"] = ica_result[:300, :2].tolist()
            
            dim_reduction = results
            print(f"[Dimensionality] Generated {len(results)} reduction methods")
        except Exception as e:
            print(f"[Dimensionality] Failed: {e}")
            dim_reduction = None

    # Feature Importance (if we can identify a target)
    feature_importance = None
    if num_df.shape[1] >= 2 and num_df.dropna().shape[0] >= 50:
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            num_clean = num_df.dropna()
            # Use last column as target, others as features
            X = num_clean.iloc[:, :-1]
            y = num_clean.iloc[:, -1]
            
            if len(X.columns) > 0:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                feature_importance = {
                    "target_column": y.name,
                    "features": X.columns.tolist(),
                    "importance_scores": rf.feature_importances_.tolist(),
                    "feature_ranking": sorted(zip(X.columns, rf.feature_importances_), 
                                            key=lambda x: x[1], reverse=True)[:10]
                }
                print(f"[Feature Importance] Analyzed {len(X.columns)} features")
        except Exception as e:
            print(f"[Feature Importance] Failed: {e}")
            feature_importance = None

    # Store all advanced analytics
    advanced_analytics = {
        "time_series": time_series_result,
        "anomaly_detection": anomaly_result,
        "clustering_analysis": clustering_analysis,
        "dimensionality_reduction": dim_reduction,
        "feature_importance": feature_importance
    }

    # Recommended charts
    rec_charts = []
    if numeric_info:       rec_charts.append({"type": "histogram",            "reason": "Distribution"})
    if correlation_matrix: rec_charts.append({"type": "correlation_heatmap",  "reason": "Relationships"})
    if pca_result:         rec_charts.append({"type": "pca_scatter",          "reason": "Dimensionality reduction"})
    if kmeans_result:      rec_charts.append({"type": "cluster_scatter",      "reason": f"Clusters k={kmeans_result['k']}"})
    if time_series_result: rec_charts.append({"type": "time_series",          "reason": "Time series decomposition"})
    if anomaly_result:     rec_charts.append({"type": "anomaly_detection",    "reason": "Outlier analysis"})
    if clustering_analysis:rec_charts.append({"type": "clustering_analysis",  "reason": "Advanced clustering"})
    if dim_reduction:      rec_charts.append({"type": "dimensionality_reduction", "reason": "Dimensionality reduction"})
    if feature_importance: rec_charts.append({"type": "feature_importance",   "reason": "Feature ranking"})
    if categorical_info:
        first = list(categorical_info.keys())[:2]
        for f in first:
            rec_charts.append({"type": "bar", "column": f, "reason": "Category freq"})

    # Use full df for meta, but all analysis is on sample
    basic = {
        "rows": int(df_full.shape[0]),
        "columns": int(df_full.shape[1]),
        "numeric_cols": int(df_full.select_dtypes(include="number").shape[1]),
        "categorical_cols": int(df_full.select_dtypes(exclude="number").shape[1])
    }

    return {
        "filename": filename,
        "profile": {"basic": basic},
        "summary": summary,
        "categorical": categorical_info,
        "numeric": numeric_info,
        "top_correlations": top_correlations,
        "correlation_matrix": correlation_matrix,
        "correlation_truncated": truncated,
        "correlation_kept_columns": kept_cols,
        "correlation_original_side": original_side,
        "pca": pca_result,
        "kmeans": kmeans_result,
        "assoc_rules": assoc_result,
        "advanced_analytics": advanced_analytics,
        "recommended_charts": rec_charts
    }

def ai_chart_description(chart_type, data_context):
    """Generate AI descriptions for different chart types"""
    if not GEMINI_MODEL:
        return {"description": f"This {chart_type} visualization shows patterns in your data."}
    
    descriptions = {
        "correlation_heatmap": "correlation patterns and relationships between variables",
        "pca_scatter": "dimensionality reduction showing data variance in 2D space",
        "kmeans_scatter": "clustering results showing natural groupings in the data",
        "value_counts": "frequency distribution of categorical values",
        "histogram": "distribution shape and data spread patterns",
        "scatter": "relationships and correlations between two variables",
        "bar_chart": "comparative values across different categories"
    }
    
    basic_desc = descriptions.get(chart_type, "data patterns and insights")
    
    try:
        prompt = f"""
You are a data visualization expert. Provide a concise 2-3 sentence description of what a {chart_type} chart reveals about data.

Context: {data_context}
Chart type: {chart_type}

Focus on:
- What patterns this chart type reveals
- How to interpret the visualization
- Key insights users should look for

Return a JSON with key "description" containing the explanation.
"""
        
        r = GEMINI_MODEL.generate_content(prompt)
        raw = (getattr(r, "text", None) or "").strip()
        
        if raw.startswith("```json"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        
        try:
            parsed = json.loads(raw)
            return parsed
        except:
            return {"description": f"This {chart_type} visualization reveals {basic_desc}. Look for patterns, trends, and relationships that can guide your analysis decisions."}
            
    except Exception as e:
        return {"description": f"This {chart_type} visualization reveals {basic_desc}. Examine the patterns to understand your data better."}

# Add endpoint for chart descriptions
@app.post("/api/chart_description")
def chart_description():
    ok_login, resp = require_login()
    if not ok_login: return resp
    
    d = request.get_json(force=True) or {}
    chart_type = d.get("chart_type", "")
    data_context = d.get("context", "dataset analysis")
    
    if not chart_type:
        return fail("chart_type required.")
    
    description = ai_chart_description(chart_type, data_context)
    return ok(**description)

# ---------- AUTO EXPLORE ----------
def ai_narrative_from_bundle(bundle):
    if not GEMINI_MODEL:
        print("[AI] Gemini model not available")
        if not GOOGLE_API_KEY:
            error_msg = "AI model not configured - missing GOOGLE_API_KEY environment variable"
            print(f"[AI] {error_msg}")
            return {"error": error_msg}
        elif not genai:
            error_msg = "AI model not configured - google-generativeai package not available"
            print(f"[AI] {error_msg}")
            return {"error": error_msg}
        else:
            error_msg = "AI model not configured - initialization failed"
            print(f"[AI] {error_msg}")
            return {"error": error_msg}
    
    # Check if bundle is None or invalid
    if not bundle or not isinstance(bundle, dict):
        error_msg = f"Invalid bundle data: {type(bundle).__name__ if bundle is not None else 'None'}"
        print(f"[AI] {error_msg}")
        return {"error": error_msg, "overview": f"AI analysis encountered an error: {error_msg}"}
    
    # Check if bundle has required structure
    if "profile" not in bundle or not isinstance(bundle.get("profile"), dict):
        error_msg = "Bundle missing profile data"
        print(f"[AI] {error_msg}")
        return {"error": error_msg, "overview": f"AI analysis encountered an error: {error_msg}"}
    
    if "basic" not in bundle["profile"]:
        error_msg = "Bundle missing basic profile data"
        print(f"[AI] {error_msg}")
        return {"error": error_msg, "overview": f"AI analysis encountered an error: {error_msg}"}
    
    try:
        # Extract advanced analytics info
        advanced = bundle.get("advanced_analytics", {})
        
        brief = {
            "basic": bundle["profile"]["basic"],
            "numeric_cols": list(bundle.get("numeric", {}).keys())[:6],
            "categorical_cols": list(bundle.get("categorical", {}).keys())[:6],
            "top_correlations": [{"a": a, "b": b, "corr": c} for a, b, c in bundle.get("top_correlations", [])[:10]],
            "pca_var": bundle.get("pca", {}).get("explained_variance") if bundle.get("pca") else None,
            "kmeans_k": bundle.get("kmeans", {}).get("k") if bundle.get("kmeans") else None,
            "rules_count": len(bundle.get("assoc_rules") or []) if bundle.get("assoc_rules") else 0,
            "time_series_available": bool(advanced.get("time_series")),
            "anomalies_detected": advanced.get("anomaly_detection", {}).get("outliers_count", 0) if advanced.get("anomaly_detection") else 0,
            "clustering_methods": len([k for k, v in advanced.get("clustering_analysis", {}).items() if v]) if advanced.get("clustering_analysis") else 0,
            "dimensionality_methods": len(advanced.get("dimensionality_reduction", {})) if advanced.get("dimensionality_reduction") else 0,
            "feature_importance_available": bool(advanced.get("feature_importance"))
        }
        
        prompt = f"""
You are an expert data scientist. Analyze this comprehensive dataset analysis and provide insights:

Dataset Info:
{json.dumps(brief, indent=2)}

This analysis includes:
- Basic statistics and correlations
- PCA dimensionality reduction
- K-means clustering
- Time series analysis (if available)
- Anomaly detection results
- Advanced clustering methods
- Multiple dimensionality reduction techniques
- Feature importance analysis

Return a JSON response with the following structure (ensure valid JSON format):
{{
  "overview": "2-3 sentence summary of the dataset characteristics and main patterns from all analyses",
  "key_findings": [
    "Finding 1 about data patterns or distributions",
    "Finding 2 about correlations or relationships", 
    "Finding 3 about clusters or groups",
    "Finding 4 about anomalies or outliers detected",
    "Finding 5 about advanced analytics insights",
    "Finding 6 about feature importance or time series patterns"
  ],
  "correlations_comment": "Insight about the correlation patterns found" or null,
  "clusters_comment": "Insight about the clustering results from multiple methods" or null,
  "pca_comment": "Insight about the PCA dimensionality reduction" or null,
  "time_series_comment": "Insight about time series patterns if available" or null,
  "anomaly_comment": "Insight about outliers and anomalies detected" or null,
  "feature_importance_comment": "Insight about feature rankings if available" or null,
  "categorical_insights": [
    "Insight 1 about categorical data patterns",
    "Insight 2 about distributions",
    "Insight 3 about potential issues"
  ],
  "advanced_insights": [
    "Insight 1 about advanced clustering patterns",
    "Insight 2 about dimensionality reduction results",
    "Insight 3 about predictive modeling potential"
  ],
  "potential_issues": [
    "Issue 1 - data quality concern",
    "Issue 2 - potential bias or limitation"
  ],
  "next_steps": [
    "Step 1 for further analysis",
    "Step 2 for data validation",
    "Step 3 for modeling or insights"
  ],
  "chart_priorities": [
    "Most important visualization type",
    "Second priority chart type", 
    "Third priority chart type"
  ]
}}

Respond with ONLY the JSON object, no other text.
"""
        
        print(f"[AI] Generating narrative for dataset with {brief.get('basic', {}).get('rows', '?')} rows")
        print(f"[AI] Brief data: {brief}")
        
        r = GEMINI_MODEL.generate_content(prompt)
        raw = (getattr(r, "text", None) or "").strip()
        
        print(f"[AI] Raw response length: {len(raw)}")
        print(f"[AI] Raw response preview: {raw[:200]}...")
        
        # Clean up common JSON formatting issues
        if raw.startswith("```json"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        elif raw.startswith("```"):
            raw = raw.replace("```", "").strip()
        
        try:
            parsed = json.loads(raw)
            print(f"[AI] Successfully parsed JSON with keys: {list(parsed.keys())}")
            return parsed
        except json.JSONDecodeError as je:
            print(f"[AI] JSON parsing failed: {je}")
            print(f"[AI] Failed content: {raw}")
            # Return a structured fallback
            fallback = {
                "overview": f"Dataset analysis completed for {brief.get('basic', {}).get('rows', '?')} rows and {brief.get('basic', {}).get('columns', '?')} columns.",
                "key_findings": [
                    f"Dataset contains {brief.get('basic', {}).get('numeric_cols', 0)} numeric and {brief.get('basic', {}).get('categorical_cols', 0)} categorical variables",
                    f"Found {len(brief.get('top_correlations', []))} significant correlations",
                    f"Clustering analysis {'completed' if brief.get('kmeans_k') else 'not available'}",
                    f"PCA analysis {'shows variance explained' if brief.get('pca_var') else 'not available'}",
                    "Analysis completed with automated insights"
                ],
                "correlations_comment": "Correlation analysis completed" if brief.get("top_correlations") else None,
                "clusters_comment": f"Identified {brief.get('kmeans_k', 'unknown')} clusters" if brief.get("kmeans_k") else None,
                "pca_comment": "Principal component analysis completed" if brief.get("pca_var") else None,
                "categorical_insights": [
                    f"Found {len(brief.get('categorical_cols', []))} categorical variables",
                    "Distribution analysis completed",
                    "Pattern recognition performed"
                ],
                "potential_issues": ["Automated analysis - review results carefully"],
                "next_steps": [
                    "Review correlation patterns",
                    "Validate clustering results", 
                    "Examine data quality"
                ],
                "chart_priorities": ["correlation_heatmap", "scatter_plots", "distribution_charts"]
            }
            print(f"[AI] Using fallback response")
            return fallback
            
    except Exception as e:
        error_msg = str(e)
        print(f"[AI] Generation failed: {e}")
        print(f"[AI] Exception type: {type(e).__name__}")
        print(f"[AI] Bundle keys: {list(bundle.keys()) if isinstance(bundle, dict) else 'Not a dict'}")
        return {"error": error_msg, "overview": f"AI analysis encountered an error: {error_msg}"}

@app.post("/api/auto_explore")
def auto_explore():
    ok_login, resp = require_login()
    if not ok_login:
        return resp

    fn = session.get("filename")
    if not fn:
        return fail("No active dataset.")

    try:
        # 1) get or build
        bundle = _get_cached_bundle(fn)
        if bundle is None:
            print(f"[Auto Explore] Building bundle for {fn}")
            bundle = build_auto_bundle(fn)
            _cache_bundle(fn, bundle)
        else:
            print(f"[Auto Explore] Using cached bundle for {fn}")

        # 2) generate AI (can use full bundle)
        print(f"[Auto Explore] Generating AI narrative...")
        ai = ai_narrative_from_bundle(bundle)
        print(f"[Auto Explore] AI result type: {type(ai)}, keys: {list(ai.keys()) if isinstance(ai, dict) else 'N/A'}")

        # 3) generate chart descriptions for all available analyses
        print(f"[Auto Explore] Generating chart descriptions...")
        chart_descriptions = generate_chart_descriptions(bundle)
        print(f"[Auto Explore] Generated {len(chart_descriptions)} chart descriptions")

        # 4) return slimmed version to client with AI descriptions
        return ok(bundle=slim_bundle(bundle), ai=ai, chart_descriptions=chart_descriptions)

    except Exception as e:
        app.logger.exception("auto_explore failed")
        return fail(f"Auto explore failed: {e}", 500)

# ---------- CORRELATION EXPORT ----------
@app.get("/api/correlation/meta")
def correlation_meta():
    ok_login, resp = require_login()
    if not ok_login: return resp
    fn = session.get("filename")
    if not fn: return fail("No active dataset.")
    try:
        df = load_df(fn)
        num = df.select_dtypes(include="number")
        side = num.shape[1]
        if side < 2:
            return ok(has=False, numeric_columns=side)
        corr_full = num.corr().fillna(0)
        _, truncated, kept_cols, orig_side = maybe_truncate_correlation(corr_full)
        return ok(has=True,
                  original_side=orig_side,
                  truncated=truncated,
                  kept_count=len(kept_cols),
                  kept_columns=kept_cols)
    except Exception as e:
        return fail(str(e), 500)

@app.get("/api/correlation/export")
def correlation_export():
    ok_login, resp = require_login()
    if not ok_login: return resp
    fn = session.get("filename")
    if not fn: return fail("No active dataset.")
    fmt = request.args.get("format", "csv").lower()
    try:
        df = load_df(fn)
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2: return fail("Not enough numeric columns.")
        corr = num.corr().fillna(0)
        if fmt == "json":
            return ok(filename=fn, correlation=corr.to_dict())
        buf = io.StringIO()
        corr.round(6).to_csv(buf)
        buf.seek(0)
        return send_file(
            io.BytesIO(buf.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{fn}_correlation.csv"
        )
    except Exception as e:
        return fail(f"Correlation export failed: {e}", 500)

@app.get("/api/correlation/png")
def correlation_png():
    ok_login, resp = require_login()
    if not ok_login: return resp
    if not MATPLOTLIB_OK:
        return fail("Matplotlib not available on server.", 500)
    fn = session.get("filename")
    if not fn: return fail("No active dataset.")
    try:
        df = load_df(fn)
        num = df.select_dtypes(include="number")
        if num.shape[1] < 2: return fail("Not enough numeric columns.")
        corr = num.corr().fillna(0)
        fig, ax = plt.subplots(figsize=(min(10, 0.45 * len(corr.columns) + 2),
                                        min(8,  0.45 * len(corr.columns) + 2)))
        cax = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        ax.set_title(f"Correlation Heatmap: {fn}", fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return send_file(
            buf,
            mimetype="image/png",
            as_attachment=True,
            download_name=f"{fn}_correlation.png"
        )
    except Exception as e:
        return fail(f"Correlation PNG failed: {e}", 500)

# ---------- REPORTS ----------
def build_markdown_report(bundle, ai, user=None):
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    fn = bundle.get("filename", "(unknown)")
    basic = bundle.get("profile", {}).get("basic", {})
    rows = basic.get("rows")
    cols = basic.get("columns")
    n_num = basic.get("numeric_cols")
    n_cat = basic.get("categorical_cols")
    rec_charts = bundle.get("recommended_charts", [])
    lines = []
    lines.append(f"# Data Exploration Report – `{fn}`\n")
    lines.append(f"**Generated:** {now}")
    if user: lines.append(f"**Requested by:** `{user}`")
    lines.append(f"**Dashboard Version:** {VERSION}\n")
    lines.append("## Dataset Metadata")
    lines.append(f"- Rows: **{rows}**")
    lines.append(f"- Columns: **{cols}**")
    lines.append(f"- Numeric Columns: **{n_num}**")
    lines.append(f"- Categorical Columns: **{n_cat}**\n")
    lines.append("### Notes")
    lines.append("- Exploratory statistics only; validate before production use.")
    lines.append("- AI narrative (if present) is heuristic and may omit context.\n")
    lines.append("## Structural Summary")
    if bundle.get("summary"):
        subset_cols = list(bundle["summary"].keys())[:10]
        lines.append(f"_Showing subset of {len(subset_cols)} columns (first 10 for brevity)._")
        for c in subset_cols:
            colinfo = bundle["summary"][c]
            top = colinfo.get("top") or colinfo.get("Top")
            uniq = colinfo.get("unique") or colinfo.get("Unique")
            lines.append(f"- **{c}**: unique={uniq} top={top}")
        lines.append("")
    lines.append("## AI Narrative")
    if ai and (ai.get("overview") or ai.get("key_findings") or ai.get("key_points")):
        if ai.get("overview"): lines.append(f"**Overview:** {ai['overview']}")
        if ai.get("key_findings"):
            lines.append("**Key Findings:**")
            for k in ai["key_findings"]:
                lines.append(f"- {k}")
        elif ai.get("key_points"):
            lines.append("**Key Points:**")
            for k in ai["key_points"]:
                lines.append(f"- {k}")
        if ai.get("correlations_comment"): lines.append(f"**Correlations:** {ai['correlations_comment']}")
        if ai.get("clusters_comment"):     lines.append(f"**Clusters:** {ai['clusters_comment']}")
        if ai.get("pca_comment"):          lines.append(f"**PCA:** {ai['pca_comment']}")
        if ai.get("next_steps"):
            lines.append("**Next Steps:**")
            for s in ai["next_steps"]:
                lines.append(f"- {s}")
        lines.append("")
    else:
        lines.append("_AI not configured or narrative unavailable._\n")
    if bundle.get("top_correlations"):
        lines.append("## Top Correlations (|r|)")
        for a, b, c in bundle["top_correlations"][:15]:
            lines.append(f"- `{a}` vs `{b}`: {round(c,4)}")
        lines.append("")
    if bundle.get("correlation_truncated"):
        lines.append(f"_Correlation matrix truncated to {len(bundle.get('correlation_kept_columns', []))} columns for dashboard performance._\n")
    if bundle.get("pca"):
        lines.append("## PCA Explained Variance")
        lines.append(", ".join([f"{round(x*100, 2)}%" for x in bundle['pca']['explained_variance']]) + "\n")
    if bundle.get("kmeans"):
        lines.append(f"## Clustering (k={bundle['kmeans']['k']})")
        lines.append("- Centers computed on numeric subset.\n")
    if bundle.get("assoc_rules"):
        lines.append("## Sample Association Rules (Top by Lift)")
        for r in bundle["assoc_rules"][:8]:
            lines.append(f"- {r['antecedents']} ⇒ {r['consequents']} (lift {round(r['lift'],3)}, conf {round(r['confidence'],3)})")
        lines.append("")
    if rec_charts:
        lines.append("## Recommended Charts")
        for rc in rec_charts:
            if isinstance(rc, dict):
                lines.append(f"- {rc.get('type')} – {rc.get('reason','')}".strip())
            else:
                lines.append(f"- {rc}")
        lines.append("")
    lines.append("---")
    lines.append(f"_Report generated by **Data Dashboard** for file `{fn}` at {now}. Version {VERSION}. For exploratory purposes only._\n")
    return "\n".join(lines)

@app.get("/api/report/markdown")
def report_markdown():
    ok_login, resp = require_login()
    if not ok_login: return resp
    fn = session.get("filename")
    if not fn: return fail("No dataset.")
    try:
        bundle = build_auto_bundle(fn)
        ai = ai_narrative_from_bundle(bundle)
        md = build_markdown_report(bundle, ai, user=session.get("user"))
        return ok(filename=fn, markdown=md)
    except Exception as e:
        return fail(f"Report generation failed: {e}", 500)

def generate_charts_for_pdf(bundle):
    """Generate chart images for PDF report inclusion with memory optimization"""
    if not MATPLOTLIB_OK:
        return {}
    
    charts = {}
    
    try:
        # Value Counts Chart
        if bundle.get("categorical"):
            for col_name, values in list(bundle["categorical"].items())[:1]:  # Just first categorical
                # Limit data points to prevent memory issues
                values_limited = values[:15]  # Max 15 categories
                
                fig, ax = plt.subplots(figsize=(8, 6))
                labels = [str(item['value'])[:20] for item in values_limited]  # Truncate labels
                counts = [item['count'] for item in values_limited]
                
                ax.bar(range(len(labels)), counts, color='#3b82f6')
                ax.set_xlabel('Categories')
                ax.set_ylabel('Count')
                ax.set_title(f'Value Counts: {col_name}')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                charts['value_counts'] = img_buffer
                plt.close()
                plt.clf()
                break  # Only do first categorical column
        
        # Correlation Heatmap
        if bundle.get("correlation_matrix"):
            corr_dict = bundle["correlation_matrix"]
            columns = list(corr_dict.keys())
            
            # Limit correlation matrix size for memory
            if len(columns) > 20:
                columns = columns[:20]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Build correlation matrix from dict
            import numpy as np
            corr_matrix = np.zeros((len(columns), len(columns)))
            for i, col1 in enumerate(columns):
                for j, col2 in enumerate(columns):
                    corr_matrix[i, j] = corr_dict.get(col1, {}).get(col2, 0)
            
            im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(columns)))
            ax.set_yticks(range(len(columns)))
            ax.set_xticklabels([col[:15] for col in columns], rotation=45, ha='right')
            ax.set_yticklabels([col[:15] for col in columns])
            ax.set_title('Correlation Heatmap')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            charts['correlation'] = img_buffer
            plt.close()
            plt.clf()
        
        # PCA Scatter Plot
        if bundle.get("pca") and bundle["pca"].get("components_2d"):
            components = bundle["pca"]["components_2d"]
            if len(components) > 0:
                # Limit points to prevent memory issues
                components_limited = components[:2000]  # Max 2000 points
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                x_vals = [comp[0] for comp in components_limited]
                y_vals = [comp[1] for comp in components_limited]
                
                ax.scatter(x_vals, y_vals, alpha=0.6, color='#3b82f6', s=20)
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                ax.set_title('PCA Scatter Plot')
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                charts['pca'] = img_buffer
                plt.close()
                plt.clf()
        
        # K-means Clustering
        if bundle.get("kmeans") and bundle["kmeans"].get("components_2d") and bundle["kmeans"].get("labels_preview"):
            components = bundle["kmeans"]["components_2d"]
            labels = bundle["kmeans"]["labels_preview"]
            
            if len(components) > 0 and len(labels) > 0:
                # Limit points to prevent memory issues
                max_points = 2000
                min_len = min(len(components), len(labels), max_points)
                components = components[:min_len]
                labels = labels[:min_len]
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                x_vals = [comp[0] for comp in components]
                y_vals = [comp[1] for comp in components]
                
                # Create scatter plot with colors by cluster
                scatter = ax.scatter(x_vals, y_vals, c=labels, alpha=0.6, s=20, cmap='viridis')
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_title(f'K-Means Clustering (k={bundle["kmeans"].get("k", "?")})')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar for clusters
                plt.colorbar(scatter, ax=ax, label='Cluster')
                
                plt.tight_layout()
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                charts['kmeans'] = img_buffer
                plt.close()
                plt.clf()
    
    except Exception as e:
        print(f"[Charts] Error generating charts: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Force garbage collection
        import gc
        gc.collect()
    
    return charts
    

def generate_comprehensive_pdf_report(bundle, ai, filename):
    """Generate a comprehensive PDF report with proper formatting, tables, and charts"""
    buffer = io.BytesIO()
    
    try:
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        styles = getSampleStyleSheet()
        normal_style = styles['Normal']
        heading_style = styles['Heading1']
        subheading_style = styles['Heading2']
        
        # Generate charts for embedding
        chart_images = generate_charts_for_pdf(bundle)
        
        # Import for chart embedding
        from reportlab.platypus import Image
        
        # Story to hold the content
        story = []
        
        # Title
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        story.append(Paragraph("Data Mining & Analysis Report", heading_style))
        story.append(Paragraph(f"Generated on {today} • Dataset: {filename}", normal_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", subheading_style))
        basic = bundle.get("profile", {}).get("basic", {})
        story.append(Paragraph(f"Dataset contains {basic.get('rows', '?'):,} rows and {basic.get('columns', '?')} columns.", normal_style))
        story.append(Paragraph(f"Numeric columns: {basic.get('numeric_cols', 0)}, Categorical columns: {basic.get('categorical_cols', 0)}", normal_style))
        story.append(Spacer(1, 15))
        
        # AI Insights
        if ai and not ai.get("error"):
            story.append(Paragraph("AI-Generated Insights", subheading_style))
            
            # Overview
            overview = ai.get("overview") or ai.get("summary", "")
            if overview:
                story.append(Paragraph("Overview", normal_style))
                story.append(Paragraph(overview, normal_style))
                story.append(Spacer(1, 10))
            
            # Key findings
            key_findings = ai.get("key_findings") or ai.get("key_points", [])
            if key_findings:
                story.append(Paragraph("Key Findings", normal_style))
                for point in key_findings:
                    story.append(Paragraph(f"• {point}", normal_style))
                story.append(Spacer(1, 10))
            
            # Additional AI insights
            if ai.get("correlations_comment"):
                story.append(Paragraph("Correlation Insights", normal_style))
                story.append(Paragraph(ai["correlations_comment"], normal_style))
                story.append(Spacer(1, 10))
            
            if ai.get("clusters_comment"):
                story.append(Paragraph("Clustering Insights", normal_style))
                story.append(Paragraph(ai["clusters_comment"], normal_style))
                story.append(Spacer(1, 10))
            
            if ai.get("next_steps"):
                story.append(Paragraph("Next Steps", normal_style))
                for step in ai["next_steps"]:
                    story.append(Paragraph(f"• {step}", normal_style))
                story.append(Spacer(1, 10))
        
        story.append(PageBreak())
        
        # Data Analysis Results
        story.append(Paragraph("Analysis Results", heading_style))
        
        # Correlation Analysis
        if bundle.get("top_correlations"):
            story.append(Paragraph("Top Correlations", subheading_style))
            
            # Add correlation chart if available
            if 'correlation' in chart_images:
                img = Image(chart_images['correlation'], width=6*72, height=4*72)  # 6x4 inches
                story.append(img)
                story.append(Spacer(1, 10))
            
            # Correlation table
            story.append(Paragraph("Strongest Correlations:", normal_style))
            for a, b, corr in bundle["top_correlations"][:10]:
                story.append(Paragraph(f"• {a} ↔ {b}: {corr:.3f}", normal_style))
            story.append(Spacer(1, 15))
        
        # PCA Analysis
        if bundle.get("pca"):
            story.append(Paragraph("Principal Component Analysis", subheading_style))
            
            # Add PCA chart if available
            if 'pca' in chart_images:
                img = Image(chart_images['pca'], width=6*72, height=4*72)
                story.append(img)
                story.append(Spacer(1, 10))
            
            # PCA explained variance
            pca = bundle["pca"]
            if pca.get("explained_variance"):
                story.append(Paragraph("Explained Variance by Component:", normal_style))
                for i, var in enumerate(pca["explained_variance"][:5], 1):
                    story.append(Paragraph(f"• PC{i}: {var:.1%}", normal_style))
            story.append(Spacer(1, 15))
        
        # K-Means Clustering
        if bundle.get("kmeans"):
            story.append(Paragraph("K-Means Clustering", subheading_style))
            
            # Add clustering chart if available
            if 'kmeans' in chart_images:
                img = Image(chart_images['kmeans'], width=6*72, height=4*72)
                story.append(img)
                story.append(Spacer(1, 10))
            
            kmeans = bundle["kmeans"]
            story.append(Paragraph(f"Optimal clusters identified: {kmeans.get('k', '?')}", normal_style))
            story.append(Spacer(1, 15))
        
        # Categorical Analysis
        if bundle.get("categorical"):
            story.append(Paragraph("Categorical Data Analysis", subheading_style))
            
            # Add value counts chart if available
            if 'value_counts' in chart_images:
                img = Image(chart_images['value_counts'], width=6*72, height=4*72)
                story.append(img)
                story.append(Spacer(1, 10))
            
            # Show top categories for first few categorical columns
            for col_name, values in list(bundle["categorical"].items())[:3]:
                story.append(Paragraph(f"Top values in {col_name}:", normal_style))
                for item in values[:5]:
                    story.append(Paragraph(f"• {item['value']}: {item['count']:,} occurrences", normal_style))
                story.append(Spacer(1, 10))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("Generated by Data Mining Dashboard • For exploratory purposes only", normal_style))
        
        # Build the PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        print(f"[PDF] Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Cleanup chart images
        for chart_buffer in chart_images.values():
            try:
                chart_buffer.close()
            except:
                pass
        
        # Force garbage collection
        import gc
        gc.collect()
        print("[PDF] Memory cleanup completed")

@app.get("/api/report/pdf")
def report_pdf():
    ok_login, resp = require_login()
    if not ok_login: return resp
    fn = session.get("filename")
    if not fn: return fail("No dataset.")
    try:
        bundle = build_auto_bundle(fn)
        ai = ai_narrative_from_bundle(bundle)
        
        # Generate comprehensive PDF
        buffer = generate_comprehensive_pdf_report(bundle, ai, fn)
        
        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"{fn}_comprehensive_report.pdf"
        )
    except Exception as e:
        return fail(f"PDF generation failed: {e}", 500)

# ---------- HEALTH ----------
@app.get("/api/health")
def health():
    return ok(
        state="healthy",
        user=session.get("user"),
        version=VERSION,
        corr_max_side=CORR_MAX_SIDE,
        corr_max_area=CORR_MAX_AREA
    )

# ---------- STATIC ----------

@app.get("/")
def root_index():
    if "user" in session:
        return redirect("/dashboard.html")
    return send_from_directory(app.static_folder, "login.html")

@app.route("/<path:anything>", methods=["OPTIONS"])
def any_options(anything):
    return ok()

@app.route("/api/<path:rest>", methods=["OPTIONS"])
def api_options(rest):
    return ok()

# ---------- MAIN ----------
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5050))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
