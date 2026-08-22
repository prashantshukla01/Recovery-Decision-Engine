"""SQLite append-only audit log system recording complete decision and execution history."""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from src.simulation.schemas import FailureContext
from src.modeling.predictor import ActionEstimate
from src.policy.schemas import Decision

DB_PATH_DEFAULT = "data/audit.db"


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
    """SQLite persistent append-only audit logger."""

    def __init__(self, db_path: str = DB_PATH_DEFAULT):
        self.db_path = db_path
        self._is_memory = (db_path == ":memory:")
        self._memory_conn: Optional[sqlite3.Connection] = None

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
        """Initializes SQLite database tables."""
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
        """Appends or updates an AuditRecord in SQLite."""
        ts_str = record.timestamp.isoformat()
        ctx_json = record.context.model_dump_json()
        est_json = json.dumps([e.model_dump() for e in record.estimates])
        dec_json = record.decision.model_dump_json()
        trans_json = json.dumps(record.state_transitions)

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
