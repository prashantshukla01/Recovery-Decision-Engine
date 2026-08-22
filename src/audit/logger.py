import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
from src.simulation.schemas import FailureContext
from src.modeling.predictor import ActionEstimate
from src.policy.schemas import Decision

logger = logging.getLogger("audit_logger")
DB_PATH_DEFAULT = "data/audit.db"

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


class AuditRecord(BaseModel):
    """Pydantic schema for full audit record for an event."""

    event_id: str = Field(..., description="Idempotency key")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: FailureContext
    estimates: List[ActionEstimate]
    decision: Decision
    outcome: Optional[Literal["success", "failure", "pending", "escalated", "aborted"]] = None
    state_transitions: List[str] = Field(default_factory=list)


class AuditLogger:
    """Persistent audit logger with hosted PostgreSQL support and local SQLite fallback."""

    def __init__(self, db_path: str = DB_PATH_DEFAULT, db_url: Optional[str] = None):
        self.db_path = db_path
        self.db_url = db_url or os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
        self.use_postgres = False
        self._is_memory = (db_path == ":memory:")
        self._memory_conn: Optional[sqlite3.Connection] = None

        if self.db_url and HAS_PSYCOPG2:
            try:
                # Test connection to Postgres
                with psycopg2.connect(self.db_url, connect_timeout=3) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                self.use_postgres = True
                logger.info("[AUDIT] Connected to hosted PostgreSQL database successfully.")
            except Exception as exc:
                logger.warning(f"audit: falling back to local SQLite storage (Postgres connection error: {exc})")
                self.use_postgres = False
        else:
            if self.db_url and not HAS_PSYCOPG2:
                logger.warning("audit: falling back to local SQLite storage (psycopg2 package absent)")
            else:
                logger.info("audit: falling back to local SQLite storage")

        if not self.use_postgres:
            if not self._is_memory:
                dirname = os.path.dirname(self.db_path)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
            else:
                self._memory_conn = sqlite3.connect(":memory:")

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory and self._memory_conn:
            return self._memory_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initializes database tables (PostgreSQL or SQLite)."""
        if self.use_postgres:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS audit_logs (
                            event_id VARCHAR(255) PRIMARY KEY,
                            timestamp VARCHAR(255) NOT NULL,
                            context_json TEXT NOT NULL,
                            estimates_json TEXT NOT NULL,
                            decision_json TEXT NOT NULL,
                            outcome VARCHAR(50),
                            transitions_json TEXT NOT NULL
                        );
                        """
                    )
                conn.commit()
        else:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    estimates_json TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    outcome TEXT,
                    transitions_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
            if not self._is_memory:
                conn.close()

    def log_record(self, record: AuditRecord):
        """Appends or updates an AuditRecord in Postgres or SQLite."""
        ts_str = record.timestamp.isoformat()
        ctx_json = record.context.model_dump_json()
        est_json = json.dumps([e.model_dump() for e in record.estimates])
        dec_json = record.decision.model_dump_json()
        trans_json = json.dumps(record.state_transitions)

        if self.use_postgres:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO audit_logs
                        (event_id, timestamp, context_json, estimates_json, decision_json, outcome, transitions_json)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (event_id) DO UPDATE SET
                            timestamp = EXCLUDED.timestamp,
                            context_json = EXCLUDED.context_json,
                            estimates_json = EXCLUDED.estimates_json,
                            decision_json = EXCLUDED.decision_json,
                            outcome = EXCLUDED.outcome,
                            transitions_json = EXCLUDED.transitions_json;
                        """,
                        (
                            record.event_id,
                            ts_str,
                            ctx_json,
                            est_json,
                            dec_json,
                            record.outcome,
                            trans_json,
                        ),
                    )
                conn.commit()
        else:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO audit_logs
                (event_id, timestamp, context_json, estimates_json, decision_json, outcome, transitions_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.event_id,
                    ts_str,
                    ctx_json,
                    est_json,
                    dec_json,
                    record.outcome,
                    trans_json,
                ),
            )
            conn.commit()
            if not self._is_memory:
                conn.close()

    def get_record(self, event_id: str) -> Optional[AuditRecord]:
        """Retrieves an AuditRecord by event_id."""
        if self.use_postgres:
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT event_id, timestamp, context_json, estimates_json, decision_json, outcome, transitions_json FROM audit_logs WHERE event_id = %s",
                        (event_id,),
                    )
                    row = cursor.fetchone()
        else:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT event_id, timestamp, context_json, estimates_json, decision_json, outcome, transitions_json FROM audit_logs WHERE event_id = ?",
                (event_id,),
            )
            row = cursor.fetchone()
            if not self._is_memory:
                conn.close()

        if not row:
            return None

        event_id, ts_str, ctx_json, est_json, dec_json, outcome, trans_json = row
        ctx = FailureContext.model_validate_json(ctx_json)
        estimates = [ActionEstimate.model_validate(e) for e in json.loads(est_json)]
        decision = Decision.model_validate_json(dec_json)
        transitions = json.loads(trans_json)
        ts = datetime.fromisoformat(ts_str)

        return AuditRecord(
            event_id=event_id,
            timestamp=ts,
            context=ctx,
            estimates=estimates,
            decision=decision,
            outcome=outcome,
            state_transitions=transitions,
        )

    def get_all_records(self, limit: int = 100) -> List[AuditRecord]:
        """Retrieves recent AuditRecords."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT event_id, timestamp, context_json, estimates_json, decision_json, outcome, transitions_json FROM audit_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        if not self._is_memory:
            conn.close()

        records = []
        for row in rows:
            event_id, ts_str, ctx_json, est_json, dec_json, outcome, trans_json = row
            ctx = FailureContext.model_validate_json(ctx_json)
            estimates = [ActionEstimate.model_validate(e) for e in json.loads(est_json)]
            decision = Decision.model_validate_json(dec_json)
            transitions = json.loads(trans_json)
            ts = datetime.fromisoformat(ts_str)

            records.append(
                AuditRecord(
                    event_id=event_id,
                    timestamp=ts,
                    context=ctx,
                    estimates=estimates,
                    decision=decision,
                    outcome=outcome,
                    state_transitions=transitions,
                )
            )
        return records
