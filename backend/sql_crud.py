"""SQLite CRUD layer for AgentSpace backend operations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence


@dataclass(frozen=True)
class QueryResult:
    rows: list[dict[str, Any]]
    rowcount: int
    lastrowid: int | None


class SqlCrudLayer:
    """Provide SQL CRUD operations backed by SQLite.

    This layer centralizes inserts, updates, deletes, and reads so future
    operations can funnel through a single interface.
    """

    def __init__(self, db_path: str = "agentspace.db") -> None:
        self.db_path = db_path
        self._ensure_connection()

    def _ensure_connection(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def execute(self, sql: str, params: Sequence[Any] | None = None) -> QueryResult:
        params = params or []
        with self._connect() as conn:
            cursor = conn.execute(sql, params)
            rows = [dict(row) for row in cursor.fetchall()]
            return QueryResult(rows=rows, rowcount=cursor.rowcount, lastrowid=cursor.lastrowid)

    def create_table(self, name: str, columns: Mapping[str, str]) -> None:
        column_sql = ", ".join(f"{col} {definition}" for col, definition in columns.items())
        sql = f"CREATE TABLE IF NOT EXISTS {name} ({column_sql})"
        self.execute(sql)

    def insert(self, table: str, data: Mapping[str, Any]) -> QueryResult:
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" for _ in data)
        sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return self.execute(sql, list(data.values()))

    def bulk_insert(self, table: str, rows: Iterable[Mapping[str, Any]]) -> QueryResult:
        rows_list = list(rows)
        if not rows_list:
            return QueryResult(rows=[], rowcount=0, lastrowid=None)
        columns = list(rows_list[0].keys())
        columns_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)
        sql = f"INSERT INTO {table} ({columns_sql}) VALUES ({placeholders})"
        values = [tuple(row[col] for col in columns) for row in rows_list]
        with self._connect() as conn:
            cursor = conn.executemany(sql, values)
            return QueryResult(rows=[], rowcount=cursor.rowcount, lastrowid=cursor.lastrowid)

    def update(
        self,
        table: str,
        data: Mapping[str, Any],
        where: str,
        params: Sequence[Any] | None = None,
    ) -> QueryResult:
        set_clause = ", ".join(f"{key} = ?" for key in data.keys())
        sql = f"UPDATE {table} SET {set_clause} WHERE {where}"
        merged_params = list(data.values()) + list(params or [])
        return self.execute(sql, merged_params)

    def delete(self, table: str, where: str, params: Sequence[Any] | None = None) -> QueryResult:
        sql = f"DELETE FROM {table} WHERE {where}"
        return self.execute(sql, list(params or []))

    def fetch_one(self, table: str, where: str, params: Sequence[Any] | None = None) -> dict[str, Any] | None:
        sql = f"SELECT * FROM {table} WHERE {where} LIMIT 1"
        result = self.execute(sql, list(params or []))
        return result.rows[0] if result.rows else None

    def fetch_all(
        self,
        table: str,
        where: str | None = None,
        params: Sequence[Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        sql = f"SELECT * FROM {table}"
        if where:
            sql += f" WHERE {where}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += " LIMIT ?"
            params = list(params or []) + [limit]
        result = self.execute(sql, list(params or []))
        return result.rows
