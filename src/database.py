"""SQLite storage for local report history."""
import json, os, sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
@contextmanager
def connect(path=None):
    db=sqlite3.connect(path or os.getenv("DATABASE_PATH","reviewmind.db"))
    try:
        with db: yield db
    finally: db.close()
def initialize(path=None):
    with connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, dataset_name TEXT NOT NULL, review_count INTEGER NOT NULL, metrics_json TEXT NOT NULL, report TEXT NOT NULL, generation_method TEXT NOT NULL, details_json TEXT NOT NULL DEFAULT '{}')")
        columns={r[1] for r in db.execute("PRAGMA table_info(analyses)")}
        if "details_json" not in columns: db.execute("ALTER TABLE analyses ADD COLUMN details_json TEXT NOT NULL DEFAULT '{}'")
def save_analysis(dataset_name,review_count,metrics,report,generation_method,path=None,details=None):
    initialize(path)
    with connect(path) as db:
        cur=db.execute("INSERT INTO analyses(created_at,dataset_name,review_count,metrics_json,report,generation_method,details_json) VALUES(?,?,?,?,?,?,?)",(datetime.now(timezone.utc).isoformat(),dataset_name,int(review_count),json.dumps(metrics),report,generation_method,json.dumps(details or {}))); return cur.lastrowid
def list_analyses(path=None):
    initialize(path)
    with connect(path) as db:
        db.row_factory=sqlite3.Row; return [dict(r) for r in db.execute("SELECT id,created_at,dataset_name,review_count,metrics_json,generation_method FROM analyses ORDER BY id DESC")]
def get_analysis(analysis_id,path=None):
    initialize(path)
    with connect(path) as db:
        db.row_factory=sqlite3.Row; row=db.execute("SELECT * FROM analyses WHERE id=?",(int(analysis_id),)).fetchone(); return dict(row) if row else None
