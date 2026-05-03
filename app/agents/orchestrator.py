"""
SROP Root Orchestrator — Google ADK agents.

Architecture:
- knowledge_agent  — product docs Q&A via RAG (uses search_docs tool)
- account_agent    — user account/build data (uses mock DB tools)
- build_root_agent — per-turn factory that injects current SessionState into
                     the root agent's instruction and wraps sub-agents as AgentTool

Routing is done by the LLM (via tool selection), NOT by string parsing.
See docs/google-adk-guide.md for the AgentTool pattern rationale.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from app.agents.tools.account_tools import get_account_status, get_recent_builds
from app.agents.tools.escalation_tools import create_ticket
from app.agents.tools.search_docs import search_docs
from app.settings import settings
from app.srop.state import SessionState

# ---------------------------------------------------------------------------
# Sub-agents — module-level singletons (no per-turn recreation overhead)
# ---------------------------------------------------------------------------

KNOWLEDGE_INSTRUCTION = """
You are the Helix product knowledge specialist.
Answer questions using ONLY the search_docs tool to retrieve documentation.
Always cite the chunk_id in your answer, e.g. "According to [chunk_abc123]...".
If the retrieved context does not contain the answer, say so — do not guess.
Return a clear, concise answer with citations.
"""

knowledge_agent = LlmAgent(
    name="knowledge_agent",
    model=settings.adk_model,
    instruction=KNOWLEDGE_INSTRUCTION,
    tools=[search_docs],
)

ACCOUNT_INSTRUCTION = """
You are the Helix account specialist.
Answer questions about builds and account status using the provided tools.
Always call the appropriate tool — do not guess values.
Return results clearly formatted for the user.
"""

account_agent = LlmAgent(
    name="account_agent",
    model=settings.adk_model,
    instruction=ACCOUNT_INSTRUCTION,
    tools=[get_recent_builds, get_account_status],
)

ESCALATION_INSTRUCTION = """
You are the Helix escalation specialist.
Handle complex issues that need human support — create tickets.
Always use create_ticket when user requests escalation or issue is unresolvable.
"""

escalation_agent = LlmAgent(
    name="escalation_agent",
    model=settings.adk_model,
    instruction=ESCALATION_INSTRUCTION,
    tools=[create_ticket],
)

# ---------------------------------------------------------------------------
# Root agent factory — injects per-turn session context into instruction
# ---------------------------------------------------------------------------

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent:

- HOW to do something, WHAT something is, docs/feature questions
  → call knowledge_agent

- Their builds, account, plan tier, usage limits
  → call account_agent

- "escalate", "support ticket", "talk to human", complex issues
  → call escalation_agent

- Greetings, thanks, or clearly off-topic messages
  → respond directly without calling a tool

IMPORTANT: When the intent matches a specialist, ALWAYS call the tool.
Never answer knowledge or account questions from memory.
"""


def build_root_agent(state: SessionState) -> LlmAgent:
    """
    Create the root orchestrator with current session context injected.

    Called once per pipeline turn. Sub-agents are module-level singletons.
    Only the root is recreated to embed fresh state — this is cheap compared
    to the LLM round-trip that follows.
    """
    context_block = (
        f"\n\nCurrent user context (do NOT re-ask for this information):\n"
        f"  user_id    : {state.user_id}\n"
        f"  plan_tier  : {state.plan_tier}\n"
        f"  last_agent : {state.last_agent or 'none'}\n"
        f"  turn_count : {state.turn_count}\n"
    )
    return LlmAgent(
        name="srop_root",
        model=settings.adk_model,
        instruction=ROOT_INSTRUCTION + context_block,
        tools=[
            AgentTool(agent=knowledge_agent),
            AgentTool(agent=account_agent),
            AgentTool(agent=escalation_agent),  # E2
        ],
    )
