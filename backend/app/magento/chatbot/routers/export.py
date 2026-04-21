"""
Server-side export endpoints for the Magento admin.

  GET  /api/magento/chatbot/agent/admin/export/chats.csv
  GET  /api/magento/chatbot/agent/admin/export/chats.pdf
  GET  /api/magento/chatbot/agent/admin/export/transcript/{session_id}.pdf

CSV is also generated client-side in the Magento module for small datasets —
the server-side CSV covers large exports and unified PDF generation. PDF uses
WeasyPrint when available; falls back to a plain-HTML attachment otherwise.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.services.database import get_db

from backend.app.magento.chatbot.db.schema import ensure_agent_schema
from backend.app.magento.chatbot.routers.common import authorize_request
from backend.app.magento.chatbot.services import chat_history_service

router = APIRouter()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _chats_csv_iter(
    db: Session,
    *,
    client_id: str,
    since: Optional[datetime],
    until: Optional[datetime],
    store_code: Optional[str],
):
    ensure_agent_schema(db)
    where = ["s.client_id = :client_id"]
    params: dict = {"client_id": client_id}
    if since:
        where.append("s.last_activity_at >= :since")
        params["since"] = since
    if until:
        where.append("s.last_activity_at <= :until")
        params["until"] = until
    if store_code:
        where.append("s.store_code = :store_code")
        params["store_code"] = store_code

    query = f"""
        SELECT s.id AS session_id, s.store_code, s.thread_id, s.customer_id, s.guest_session_id,
               s.is_customer_login, s.started_at, s.last_activity_at, s.message_count,
               m.id AS message_id, m.role, m.agent_name, m.tool_name, m.content,
               m.input_tokens, m.output_tokens, m.total_tokens, m.total_cost,
               m.response_time_ms, m.created_at
        FROM agent_chat_sessions s
        LEFT JOIN agent_chat_messages m ON m.session_id = s.id
        WHERE {" AND ".join(where)}
        ORDER BY s.last_activity_at DESC, m.created_at ASC
    """

    headers = [
        "session_id", "store_code", "thread_id",
        "customer_id", "guest_session_id", "is_customer_login",
        "session_started_at", "session_last_activity_at", "session_message_count",
        "message_id", "role", "agent_name", "tool_name", "content",
        "input_tokens", "output_tokens", "total_tokens", "total_cost",
        "response_time_ms", "message_created_at",
    ]

    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL)
    writer.writerow(headers)
    yield buffer.getvalue()
    buffer.seek(0)
    buffer.truncate(0)

    result = db.execute(text(query), params)
    for row in result:
        writer.writerow([
            row.session_id, row.store_code, row.thread_id,
            row.customer_id or "", row.guest_session_id or "",
            int(row.is_customer_login or 0),
            row.started_at.isoformat() if row.started_at else "",
            row.last_activity_at.isoformat() if row.last_activity_at else "",
            int(row.message_count or 0),
            row.message_id or "", row.role or "", row.agent_name or "", row.tool_name or "",
            (row.content or "").replace("\n", "\\n"),
            int(row.input_tokens or 0), int(row.output_tokens or 0),
            int(row.total_tokens or 0), float(row.total_cost or 0),
            int(row.response_time_ms or 0),
            row.created_at.isoformat() if row.created_at else "",
        ])
        yield buffer.getvalue()
        buffer.seek(0)
        buffer.truncate(0)


@router.get("/magento/chatbot/agent/admin/export/chats.csv")
def export_chats_csv(
    request: Request,
    since: Optional[str] = Query(None),
    until: Optional[str] = Query(None),
    store_code: Optional[str] = Query(None),
    days: Optional[int] = Query(None, ge=1, le=365),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )

    since_dt = _parse_iso(since)
    until_dt = _parse_iso(until)
    if days and not since_dt:
        since_dt = datetime.utcnow() - timedelta(days=days)

    filename = f"chatbot_chats_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        _chats_csv_iter(
            db,
            client_id=license_data["client_id"],
            since=since_dt, until=until_dt, store_code=store_code,
        ),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── PDF ──────────────────────────────────────────────────────────────────────


def _render_html_chats(rows: list, client_name: str, since, until) -> str:
    style = "<style>body{font-family:sans-serif;font-size:11px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:4px;text-align:left}th{background:#eee}</style>"
    header = f"<h2>Chat History — {client_name}</h2><p>From {since or 'beginning'} to {until or 'now'}</p>"
    table = (
        "<table><thead><tr>"
        "<th>Date</th><th>Session</th><th>Role</th><th>Agent</th><th>Tokens</th><th>Content</th>"
        "</tr></thead><tbody>"
    )
    for r in rows:
        table += (
            "<tr>"
            f"<td>{(r.created_at or '')}</td>"
            f"<td>{r.session_id}</td>"
            f"<td>{r.role or ''}</td>"
            f"<td>{r.agent_name or ''}</td>"
            f"<td>{int(r.total_tokens or 0)}</td>"
            f"<td>{(r.content or '')[:400].replace('<', '&lt;')}</td>"
            "</tr>"
        )
    table += "</tbody></table>"
    return f"<html><head>{style}</head><body>{header}{table}</body></html>"


@router.get("/magento/chatbot/agent/admin/export/chats.pdf")
def export_chats_pdf(
    request: Request,
    since: Optional[str] = Query(None),
    until: Optional[str] = Query(None),
    store_code: Optional[str] = Query(None),
    days: Optional[int] = Query(30, ge=1, le=365),
    limit: int = Query(1000, ge=1, le=5000),
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    ensure_agent_schema(db)

    since_dt = _parse_iso(since)
    until_dt = _parse_iso(until)
    if days and not since_dt:
        since_dt = datetime.utcnow() - timedelta(days=days)

    where = ["s.client_id = :client_id"]
    params: dict = {"client_id": license_data["client_id"], "limit": limit}
    if since_dt:
        where.append("m.created_at >= :since")
        params["since"] = since_dt
    if until_dt:
        where.append("m.created_at <= :until")
        params["until"] = until_dt
    if store_code:
        where.append("s.store_code = :store_code")
        params["store_code"] = store_code

    rows = db.execute(
        text(
            f"""
            SELECT m.id AS message_id, m.session_id, m.role, m.agent_name, m.tool_name,
                   m.content, m.total_tokens, m.created_at
            FROM agent_chat_messages m
            JOIN agent_chat_sessions s ON s.id = m.session_id
            WHERE {" AND ".join(where)}
            ORDER BY m.created_at DESC
            LIMIT :limit
            """
        ),
        params,
    ).fetchall()

    html = _render_html_chats(rows, license_data.get("client_name") or license_data["client_id"], since_dt, until_dt)

    try:
        from weasyprint import HTML  # type: ignore

        pdf_bytes = HTML(string=html).write_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="chatbot_chats_{datetime.utcnow().strftime("%Y%m%d")}.pdf"'
            },
        )
    except Exception:
        # WeasyPrint unavailable — return the raw HTML so the admin still gets a usable export.
        return Response(
            content=html,
            media_type="text/html",
            headers={
                "Content-Disposition": f'attachment; filename="chatbot_chats_{datetime.utcnow().strftime("%Y%m%d")}.html"'
            },
        )


@router.get("/magento/chatbot/agent/admin/export/transcript/{session_id}.pdf")
def export_transcript_pdf(
    session_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    license_data = authorize_request(
        request=request, db=db,
        authorization=authorization, x_api_key=x_api_key,
        request_license=None,
    )
    session = chat_history_service.get_session(
        db, client_id=license_data["client_id"], session_id=session_id
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = chat_history_service.get_session_messages(
        db, client_id=license_data["client_id"], session_id=session_id
    )

    style = "<style>body{font-family:sans-serif;font-size:12px}.msg{margin:8px 0;padding:8px;border-radius:6px}.user{background:#eef}.assistant{background:#efe}.tool{background:#fafafa;font-family:monospace;font-size:10px}</style>"
    lines = [f"<h2>Transcript — session {session_id}</h2>"]
    lines.append(f"<p>Store: {session.get('store_code')} · Customer: {session.get('customer_id') or 'guest'} · "
                 f"Thread: {session.get('thread_id')}</p>")
    for m in messages:
        role = m.get("role") or "?"
        content = (m.get("content") or "").replace("<", "&lt;").replace("\n", "<br>")
        lines.append(
            f'<div class="msg {role}"><strong>{role}{" (" + (m.get("agent_name") or "") + ")" if m.get("agent_name") else ""}:</strong><br>{content}</div>'
        )
    html = f"<html><head>{style}</head><body>{''.join(lines)}</body></html>"

    try:
        from weasyprint import HTML  # type: ignore

        pdf_bytes = HTML(string=html).write_pdf()
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="transcript_{session_id}.pdf"'
            },
        )
    except Exception:
        return Response(
            content=html,
            media_type="text/html",
            headers={
                "Content-Disposition": f'attachment; filename="transcript_{session_id}.html"'
            },
        )
