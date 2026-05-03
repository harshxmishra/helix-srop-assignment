"""
Account tools — used by AccountAgent.

Returns mock data. The assignment spec says mock is fine;
what's evaluated is that these functions are wired as ADK tools
and the agent invokes them when the user asks about builds/account.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class BuildSummary:
    build_id: str
    pipeline: str
    status: str  # passed | failed | cancelled
    branch: str
    started_at: str   # ISO string (dataclasses with datetime don't serialise well for ADK)
    duration_seconds: int


@dataclass
class AccountStatus:
    user_id: str
    plan_tier: str
    concurrent_builds_used: int
    concurrent_builds_limit: int
    storage_used_gb: float
    storage_limit_gb: float


# Static mock builds — deterministic so tests are stable
_MOCK_BUILDS = [
    BuildSummary("build_001", "ci.yml", "passed",    "main",    "2026-05-03T10:00:00Z", 120),
    BuildSummary("build_002", "ci.yml", "failed",    "feature/x","2026-05-03T09:30:00Z", 45),
    BuildSummary("build_003", "ci.yml", "passed",    "main",    "2026-05-02T18:00:00Z", 135),
    BuildSummary("build_004", "deploy.yml", "cancelled", "hotfix", "2026-05-02T14:00:00Z", 10),
    BuildSummary("build_005", "ci.yml", "passed",    "main",    "2026-05-01T12:00:00Z", 118),
]

_PLAN_LIMITS = {
    "free":       (1, 5.0),
    "pro":        (5, 50.0),
    "enterprise": (20, 500.0),
}


async def get_recent_builds(user_id: str, limit: int = 5) -> list[dict]:
    """
    Return the most recent builds for a user, newest first.

    Returns a list of dicts (JSON-serialisable) so ADK can relay the result
    back to the root agent cleanly.
    """
    builds = _MOCK_BUILDS[:limit]
    return [asdict(b) for b in builds]


async def get_account_status(user_id: str) -> dict:
    """
    Return current account status including plan tier and usage limits.

    Returns a dict so ADK can relay it as a tool result.
    """
    # Derive plan_tier from user_id suffix for demo variety (default: free)
    tier = "free"
    for t in ("pro", "enterprise"):
        if t in user_id.lower():
            tier = t
            break

    concurrent_limit, storage_limit = _PLAN_LIMITS.get(tier, (1, 5.0))
    status = AccountStatus(
        user_id=user_id,
        plan_tier=tier,
        concurrent_builds_used=1,
        concurrent_builds_limit=concurrent_limit,
        storage_used_gb=2.3,
        storage_limit_gb=storage_limit,
    )
    return asdict(status)
