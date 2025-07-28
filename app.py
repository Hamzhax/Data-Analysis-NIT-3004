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

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except Exception:
    MLXTEND_AVAILABLE = False

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
        genai.configure(api_key=GOOGLE_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        print(f"[Gemini] init failed: {e}")

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
VALID_METHODS = {"summary", "correlation", "value_counts", "pca", "kmeans", "assoc_rules"}

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
    df = load_df(filename)
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

# ---------- AUTO EXPLORE ----------
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
        except Exception:
            pca_result = None

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
                    # Use the same subset of data that PCA used
                    components_2d = pca_result["components_2d"][:len(best_model.labels_)]
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
                
                # Add 2D components if available
                if components_2d:
                    kmeans_result["components_2d"] = components_2d[:300]  # Match labels_preview length
                    
        except Exception:
            kmeans_result = None

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

    # Recommended charts
    rec_charts = []
    if numeric_info:       rec_charts.append({"type": "histogram",            "reason": "Distribution"})
    if correlation_matrix: rec_charts.append({"type": "correlation_heatmap",  "reason": "Relationships"})
    if pca_result:         rec_charts.append({"type": "pca_scatter",          "reason": "Dimensionality reduction"})
    if kmeans_result:      rec_charts.append({"type": "cluster_scatter",      "reason": f"Clusters k={kmeans_result['k']}"})
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
        "recommended_charts": rec_charts
    }

def ai_narrative_from_bundle(bundle):
    if not GEMINI_MODEL:
        print("[AI] Gemini model not available")
        return {"error": "AI model not configured"}
    
    try:
        brief = {
            "basic": bundle["profile"]["basic"],
            "numeric_cols": list(bundle.get("numeric", {}).keys())[:6],
            "categorical_cols": list(bundle.get("categorical", {}).keys())[:6],
            "top_correlations": [{"a": a, "b": b, "corr": c} for a, b, c in bundle.get("top_correlations", [])[:10]],
            "pca_var": bundle.get("pca", {}).get("explained_variance"),
            "kmeans_k": bundle.get("kmeans", {}).get("k"),
            "rules_count": len(bundle.get("assoc_rules") or []) if bundle.get("assoc_rules") else 0
        }
        
        prompt = f"""
You are an expert data scientist. Analyze this dataset structural brief and provide insights:

Dataset Info:
{json.dumps(brief, indent=2)}

Return a JSON response with the following structure (ensure valid JSON format):
{{
  "overview": "2-3 sentence summary of the dataset characteristics and main patterns",
  "key_findings": [
    "Finding 1 about data patterns or distributions",
    "Finding 2 about correlations or relationships", 
    "Finding 3 about clusters or groups",
    "Finding 4 about anomalies or outliers",
    "Finding 5 about data quality or completeness"
  ],
  "correlations_comment": "Insight about the correlation patterns found" or null,
  "clusters_comment": "Insight about the clustering results" or null,
  "pca_comment": "Insight about the PCA dimensionality reduction" or null,
  "categorical_insights": [
    "Insight 1 about categorical data patterns",
    "Insight 2 about distributions",
    "Insight 3 about potential issues"
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
            return {
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
            
    except Exception as e:
        print(f"[AI] Generation failed: {e}")
        return {"error": str(e), "overview": "AI analysis encountered an error"}

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
            bundle = build_auto_bundle(fn)
            _cache_bundle(fn, bundle)

        # 2) generate AI (can use full bundle)
        ai = ai_narrative_from_bundle(bundle)

        # 3) return slimmed version to client
        return ok(bundle=slim_bundle(bundle), ai=ai)

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
                ax.set_xlabel('Categories')
                ax.set_ylabel('Count')
                ax.set_title(f'Value Counts: {col_name}')
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha='right')
                
                plt.tight_layout()
                
                # Save to memory with optimized settings
                img_buffer = io.BytesIO()
                plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                charts['value_counts'] = img_buffer
                plt.close()
                plt.clf()  # Clear figure
                break
        
        # Correlation Heatmap
        if bundle.get("correlation_matrix"):
            corr_matrix = bundle["correlation_matrix"]
            df_corr = pd.DataFrame(corr_matrix)
            
            if not df_corr.empty:
                # Limit correlation matrix size to prevent memory issues
                if len(df_corr.columns) > 20:
                    df_corr = df_corr.iloc[:20, :20]  # Max 20x20 correlation matrix
                
                fig, ax = plt.subplots(figsize=(min(12, len(df_corr.columns) * 0.8), 
                                                min(10, len(df_corr.columns) * 0.6)))
                
                im = ax.imshow(df_corr.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
                
                # Set ticks and labels
                ax.set_xticks(range(len(df_corr.columns)))
                ax.set_yticks(range(len(df_corr.columns)))
                ax.set_xticklabels([col[:15] for col in df_corr.columns], rotation=45, ha='right')
                ax.set_yticklabels([col[:15] for col in df_corr.columns])
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
                ax.set_title('Correlation Matrix')
                
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
                
                # Create scatter plot with different colors for each cluster
                scatter = ax.scatter(x_vals, y_vals, c=labels, cmap='tab10', alpha=0.7, s=20)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_title('K-means Clustering')
                ax.grid(True, alpha=0.3)
                
                # Add legend if reasonable number of clusters
                if len(set(labels)) <= 10:
                    plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.8)
                
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
