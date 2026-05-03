"""
E2 Escalation Agent tools.
create_ticket — open support ticket, stored in DB.
"""
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class TicketSummary:
    ticket_id: str
    summary: str
    priority: str
    status: str
    created_at: str


async def create_ticket(
    user_id: str,
    session_id: str,
    summary: str,
    priority: str = "medium",
) -> dict:
    """
    Create support ticket. Store in DB (passed via session context).

    Args:
        user_id: who is creating ticket
        session_id: session context (for linking)
        summary: issue description
        priority: low|medium|high|urgent

    Returns:
        dict with ticket_id, summary, priority, status (JSON-serializable)
    """
    # Note: actual DB write happens in pipeline.py context
    # This function is called by escalation_agent, returns data
    # Pipeline writes to Ticket table
    ticket_id = f"ticket_{str(uuid.uuid4())[:8]}"
    ticket = TicketSummary(
        ticket_id=ticket_id,
        summary=summary,
        priority=priority,
        status="open",
        created_at=datetime.utcnow().isoformat(),
    )
    return asdict(ticket)
