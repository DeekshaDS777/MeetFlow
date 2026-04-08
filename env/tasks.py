from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional


TASK_BANK: Dict[str, List[Dict]] = {
    "easy": [
        {
            "id": "easy_1",
            "transcript": (
                "Sprint planning: Ashu should refresh the onboarding guide by Wednesday. "
                "Divya will prepare the customer demo checklist by Thursday. "
                "Priya needs to verify the production backup today."
            ),
            "meeting_context": {
                "meeting_type": "planning",
                "team": "product_ops",
                "urgency": "normal",
                "participants": ["Ashu", "Divya", "Priya", "Sanjai"],
            },
            "ground_truth": [
                {
                    "title": "refresh onboarding guide",
                    "owner": "Ashu",
                    "priority": "medium",
                    "deadline": "Wednesday",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "prepare customer demo checklist",
                    "owner": "Divya",
                    "priority": "medium",
                    "deadline": "Thursday",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "verify production backup",
                    "owner": "Priya",
                    "priority": "high",
                    "deadline": "today",
                    "dependency": "none",
                    "risk_flag": True,
                },
            ],
        },
        {
            "id": "easy_2",
            "transcript": (
                "Weekly ops sync: Meena should update the release notes by Friday. "
                "Ashu will retest the signup flow tomorrow. "
                "Divya needs to archive the stale feature flags today."
            ),
            "meeting_context": {
                "meeting_type": "ops_sync",
                "team": "app_ops",
                "urgency": "normal",
                "participants": ["Meena", "Ashu", "Divya", "Priya"],
            },
            "ground_truth": [
                {
                    "title": "update release notes",
                    "owner": "Meena",
                    "priority": "medium",
                    "deadline": "Friday",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "retest signup flow",
                    "owner": "Ashu",
                    "priority": "medium",
                    "deadline": "tomorrow",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "archive stale feature flags",
                    "owner": "Divya",
                    "priority": "medium",
                    "deadline": "today",
                    "dependency": "none",
                    "risk_flag": False,
                },
            ],
        },
        {
            "id": "easy_3",
            "transcript": (
                "Documentation review: Rohit should refresh the API quickstart by Thursday. "
                "Priya will verify the billing FAQ links today. "
                "Sanjai needs to prepare the support handoff note by end of day."
            ),
            "meeting_context": {
                "meeting_type": "documentation_review",
                "team": "support_enablement",
                "urgency": "normal",
                "participants": ["Rohit", "Priya", "Sanjai", "Meena"],
            },
            "ground_truth": [
                {
                    "title": "refresh API quickstart",
                    "owner": "Rohit",
                    "priority": "medium",
                    "deadline": "Thursday",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "verify billing FAQ links",
                    "owner": "Priya",
                    "priority": "medium",
                    "deadline": "today",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "prepare support handoff note",
                    "owner": "Sanjai",
                    "priority": "medium",
                    "deadline": "end of day",
                    "dependency": "none",
                    "risk_flag": False,
                },
            ],
        },
    ],
    "medium": [
        {
            "id": "medium_1",
            "transcript": (
                "Incident review: Sanjai should reproduce the payment timeout today. "
                "Meena will draft the RCA summary by tomorrow after Sanjai reproduces the issue. "
                "Vikram must finish the migration checklist tonight because rollout depends on it."
            ),
            "meeting_context": {
                "meeting_type": "incident_review",
                "team": "platform",
                "urgency": "high",
                "participants": ["Sanjai", "Meena", "Vikram", "Priya"],
            },
            "ground_truth": [
                {
                    "title": "reproduce payment timeout",
                    "owner": "Sanjai",
                    "priority": "high",
                    "deadline": "today",
                    "dependency": "none",
                    "risk_flag": True,
                },
                {
                    "title": "draft RCA summary",
                    "owner": "Meena",
                    "priority": "high",
                    "deadline": "tomorrow",
                    "dependency": "reproduce payment timeout",
                    "risk_flag": False,
                },
                {
                    "title": "finish migration checklist",
                    "owner": "Vikram",
                    "priority": "high",
                    "deadline": "tonight",
                    "dependency": "none",
                    "risk_flag": True,
                },
            ],
        },
        {
            "id": "medium_2",
            "transcript": (
                "Release gate meeting: Kousik must approve rollout communication today. "
                "Ashu should verify the production backup before rollout. "
                "Divya will prepare the rollback checklist after Ashu confirms the backup."
            ),
            "meeting_context": {
                "meeting_type": "release_gate",
                "team": "release_ops",
                "urgency": "high",
                "participants": ["Kousik", "Ashu", "Divya", "Priya"],
            },
            "ground_truth": [
                {
                    "title": "approve rollout communication",
                    "owner": "Kousik",
                    "priority": "high",
                    "deadline": "today",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "verify production backup",
                    "owner": "Ashu",
                    "priority": "high",
                    "deadline": "before rollout",
                    "dependency": "none",
                    "risk_flag": True,
                },
                {
                    "title": "prepare rollback checklist",
                    "owner": "Divya",
                    "priority": "high",
                    "deadline": "after backup confirmation",
                    "dependency": "verify production backup",
                    "risk_flag": True,
                },
            ],
        },
        {
            "id": "medium_3",
            "transcript": (
                "Customer escalation sync: Priya should confirm the enterprise workaround by today. "
                "Rohit will prepare the follow-up summary after Priya confirms the workaround. "
                "Meena must validate dashboard freshness tonight because stale metrics are still being reported."
            ),
            "meeting_context": {
                "meeting_type": "customer_escalation",
                "team": "customer_ops",
                "urgency": "high",
                "participants": ["Priya", "Rohit", "Meena", "Sanjai"],
            },
            "ground_truth": [
                {
                    "title": "confirm enterprise workaround",
                    "owner": "Priya",
                    "priority": "high",
                    "deadline": "today",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "prepare follow-up summary",
                    "owner": "Rohit",
                    "priority": "medium",
                    "deadline": "after workaround confirmation",
                    "dependency": "confirm enterprise workaround",
                    "risk_flag": False,
                },
                {
                    "title": "validate dashboard freshness",
                    "owner": "Meena",
                    "priority": "high",
                    "deadline": "tonight",
                    "dependency": "none",
                    "risk_flag": True,
                },
            ],
        },
    ],
    "hard": [
        {
            "id": "hard_1",
            "transcript": (
                "Security escalation: Priya must revoke the compromised access token immediately. "
                "Rohit should rotate the affected service credentials within one hour after Priya revokes the token. "
                "Divya needs to prepare the internal incident brief before 5 PM. "
                "Sanjai will validate the credential rotation before rollout. "
                "Kousik should update the rollback plan if validation fails."
            ),
            "meeting_context": {
                "meeting_type": "security_escalation",
                "team": "security_ops",
                "urgency": "critical",
                "participants": ["Priya", "Rohit", "Divya", "Sanjai", "Kousik"],
            },
            "ground_truth": [
                {
                    "title": "revoke compromised access token",
                    "owner": "Priya",
                    "priority": "critical",
                    "deadline": "immediately",
                    "dependency": "none",
                    "risk_flag": True,
                },
                {
                    "title": "rotate affected service credentials",
                    "owner": "Rohit",
                    "priority": "critical",
                    "deadline": "within one hour",
                    "dependency": "revoke compromised access token",
                    "risk_flag": True,
                },
                {
                    "title": "prepare internal incident brief",
                    "owner": "Divya",
                    "priority": "high",
                    "deadline": "before 5 PM",
                    "dependency": "none",
                    "risk_flag": False,
                },
                {
                    "title": "validate credential rotation",
                    "owner": "Sanjai",
                    "priority": "high",
                    "deadline": "before rollout",
                    "dependency": "rotate affected service credentials",
                    "risk_flag": True,
                },
                {
                    "title": "update rollback plan",
                    "owner": "Kousik",
                    "priority": "high",
                    "deadline": "if validation fails",
                    "dependency": "validate credential rotation",
                    "risk_flag": True,
                },
            ],
        },
        {
            "id": "hard_2",
            "transcript": (
                "Launch war room: Kousik must hotfix the API timeout before 3 PM because checkout is failing for premium users. "
                "Ashu will send the enterprise customer update right after Kousik confirms the hotfix. "
                "Divya should update the rollback runbook by end of day in case the hotfix fails. "
                "Sanjai owns validation of the hotfix on staging before rollout. "
                "Priya needs to approve the rollback trigger conditions before launch."
            ),
            "meeting_context": {
                "meeting_type": "launch_war_room",
                "team": "cross_functional",
                "urgency": "critical",
                "participants": ["Kousik", "Ashu", "Divya", "Sanjai", "Priya"],
            },
            "ground_truth": [
                {
                    "title": "hotfix API timeout",
                    "owner": "Kousik",
                    "priority": "critical",
                    "deadline": "before 3 PM",
                    "dependency": "none",
                    "risk_flag": True,
                },
                {
                    "title": "send enterprise customer update",
                    "owner": "Ashu",
                    "priority": "high",
                    "deadline": "after hotfix confirmation",
                    "dependency": "hotfix API timeout",
                    "risk_flag": False,
                },
                {
                    "title": "update rollback runbook",
                    "owner": "Divya",
                    "priority": "high",
                    "deadline": "end of day",
                    "dependency": "none",
                    "risk_flag": True,
                },
                {
                    "title": "validate hotfix on staging",
                    "owner": "Sanjai",
                    "priority": "high",
                    "deadline": "before rollout",
                    "dependency": "hotfix API timeout",
                    "risk_flag": True,
                },
                {
                    "title": "approve rollback trigger conditions",
                    "owner": "Priya",
                    "priority": "high",
                    "deadline": "before launch",
                    "dependency": "update rollback runbook",
                    "risk_flag": True,
                },
            ],
        },
        {
            "id": "hard_3",
            "transcript": (
                "Release recovery meeting: Vikram must rebuild the failed analytics pipeline before noon because dashboards for finance are stale. "
                "Meena should publish the stakeholder update after Vikram confirms the rebuild. "
                "Ashu needs to validate the corrected metrics before finance reviews them. "
                "Rohit will prepare the rollback snapshot in case validation fails. "
                "Divya must archive the incident notes by end of day."
            ),
            "meeting_context": {
                "meeting_type": "release_recovery",
                "team": "data_platform",
                "urgency": "critical",
                "participants": ["Vikram", "Meena", "Ashu", "Rohit", "Divya"],
            },
            "ground_truth": [
                {
                    "title": "rebuild failed analytics pipeline",
                    "owner": "Vikram",
                    "priority": "critical",
                    "deadline": "before noon",
                    "dependency": "none",
                    "risk_flag": True,
                },
                {
                    "title": "publish stakeholder update",
                    "owner": "Meena",
                    "priority": "high",
                    "deadline": "after rebuild confirmation",
                    "dependency": "rebuild failed analytics pipeline",
                    "risk_flag": False,
                },
                {
                    "title": "validate corrected metrics",
                    "owner": "Ashu",
                    "priority": "high",
                    "deadline": "before finance review",
                    "dependency": "rebuild failed analytics pipeline",
                    "risk_flag": True,
                },
                {
                    "title": "prepare rollback snapshot",
                    "owner": "Rohit",
                    "priority": "high",
                    "deadline": "if validation fails",
                    "dependency": "validate corrected metrics",
                    "risk_flag": True,
                },
                {
                    "title": "archive incident notes",
                    "owner": "Divya",
                    "priority": "medium",
                    "deadline": "end of day",
                    "dependency": "none",
                    "risk_flag": False,
                },
            ],
        },
    ],
}


def get_task(task_name: str, episode_idx: Optional[int] = None) -> Dict:
    if task_name not in TASK_BANK:
        raise KeyError(f"Unknown task difficulty: {task_name}")

    bank = TASK_BANK[task_name]
    if not bank:
        raise ValueError(f"No tasks configured for difficulty: {task_name}")

    if episode_idx is None:
        task = bank[0]
    else:
        task = bank[(episode_idx - 1) % len(bank)]

    return deepcopy(task)


def list_task_ids(task_name: str) -> List[str]:
    if task_name not in TASK_BANK:
        return []
    return [task["id"] for task in TASK_BANK[task_name]]