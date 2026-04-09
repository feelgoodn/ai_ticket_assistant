import re
import requests
from datetime import datetime
from typing import Dict, Optional

from agent.config import (
    SNOW_INSTANCE,
    SNOW_USERNAME,
    SNOW_PASSWORD,
    SNOW_TIMEOUT,
    PRIORITY_LABEL_MAP,
    PRIORITY_IMPACT_URGENCY,
    TICKET_STATE_MAP,
    STATE_NAME_MAP,
)

class ServiceNowClient:

    _session: Optional[requests.Session] = None
    _user_cache: Dict[str, str] = {}
    _sys_id_cache: Dict[str, str] = {}

    def __init__(self):
        if not all([SNOW_INSTANCE, SNOW_USERNAME, SNOW_PASSWORD]):
            raise EnvironmentError(
                "SERVICENOW_INSTANCE, SERVICENOW_USERNAME, and SERVICENOW_PASSWORD must be set."
            )
        self.base_url = f"https://{SNOW_INSTANCE}/api/now/table/incident"
        self.user_url = f"https://{SNOW_INSTANCE}/api/now/table/sys_user"

        if ServiceNowClient._session is None:
            s = requests.Session()
            s.auth = (SNOW_USERNAME, SNOW_PASSWORD)
            s.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
            s.mount("https://", requests.adapters.HTTPAdapter(
                pool_connections=5, pool_maxsize=10,
                max_retries=requests.adapters.Retry(
                    total=1, backoff_factor=0.2, status_forcelist=[500, 502, 503, 504]
                ),
            ))
            ServiceNowClient._session = s

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_dt(dt_str: str) -> str:
        if not dt_str:
            return "N/A"
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").strftime("%b %d, %Y at %I:%M %p")
        except ValueError:
            return dt_str

    def _get_sys_id(self, ticket_number: str) -> Optional[str]:
        key = ticket_number.upper()
        if key in ServiceNowClient._sys_id_cache:
            return ServiceNowClient._sys_id_cache[key]
        resp = ServiceNowClient._session.get(
            self.base_url,
            params={"sysparm_query": f"number={key}", "sysparm_fields": "sys_id", "sysparm_limit": 1},
            timeout=SNOW_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("result", [])
        sys_id = results[0].get("sys_id") if results else None
        if sys_id:
            ServiceNowClient._sys_id_cache[key] = sys_id
        return sys_id

    def _lookup_caller(self, email: str) -> Optional[str]:
        key = email.lower()
        if key in ServiceNowClient._user_cache:
            return ServiceNowClient._user_cache[key]
        try:
            resp = ServiceNowClient._session.get(
                self.user_url,
                params={"sysparm_query": f"email={email}", "sysparm_fields": "sys_id", "sysparm_limit": 1},
                timeout=SNOW_TIMEOUT,
            )
            if resp.status_code == 200:
                results = resp.json().get("result", [])
                if results:
                    sys_id = results[0]["sys_id"]
                    ServiceNowClient._user_cache[key] = sys_id
                    return sys_id
        except Exception as exc:
            print(f"[tools] Caller lookup error: {exc}")
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def create_ticket(self, name: str, contact: str, description: str, priority: str = "medium") -> Dict:
        p_key = (priority or "medium").strip().lower()
        if p_key == "critical":
            p_key = "high"
        impact, urgency = PRIORITY_IMPACT_URGENCY.get(p_key, PRIORITY_IMPACT_URGENCY["medium"])
        short_desc = description[:100] + ("..." if len(description) > 100 else "")

        caller_id = self._lookup_caller(contact) if "@" in contact else None

        payload: Dict = {
            "short_description": short_desc,
            "description":       description,
            "impact":            str(impact),
            "urgency":           str(urgency),
            "contact_type":      "self-service",
        }
        if caller_id:
            payload["caller_id"] = caller_id

        try:
            resp = ServiceNowClient._session.post(self.base_url, json=payload, timeout=SNOW_TIMEOUT)
            resp.raise_for_status()
            result = resp.json().get("result") or {}
            number = result.get("number")
            if not number:
                return {"success": False, "message": "❌ ServiceNow did not return a ticket number."}
            sys_id = result.get("sys_id")
            if sys_id:
                ServiceNowClient._sys_id_cache[number] = sys_id
                # Store reporter info as a work note (keeps description clean)
                try:
                    ServiceNowClient._session.patch(
                        f"{self.base_url}/{sys_id}",
                        json={"work_notes": f"Reported by: {name} ({contact})"},
                        timeout=SNOW_TIMEOUT,
                    )
                except Exception:
                    pass
        except Exception as exc:
            return {"success": False, "message": f"❌ Failed to create ticket: {exc}"}

        _PRIORITY_DISPLAY = {"high": "High (Critical)", "medium": "Medium", "low": "Low"}
        priority_display = _PRIORITY_DISPLAY.get(p_key, p_key.capitalize())
        return {
            "success":       True,
            "ticket_number": number,
            "priority":      priority_display,
            "message": (
                f"✅ Ticket **{number}** was created successfully.\n"
                f"• Priority: **{priority_display}**\n"
                f"• Issue: {short_desc}\n\n"
                f"Our support team will review your issue and keep you updated. "
                f"Please save your ticket number for future reference."
            ),
        }

    def get_ticket_status(self, ticket_number: str) -> Dict:
        try:
            resp = ServiceNowClient._session.get(
                self.base_url,
                params={
                    "sysparm_query":         f"number={ticket_number.upper()}",
                    "sysparm_fields":        "number,sys_id,state,sys_updated_on,description,priority",
                    "sysparm_limit":         1,
                    "sysparm_display_value": "all",
                },
                timeout=SNOW_TIMEOUT,
            )
            resp.raise_for_status()
            results = resp.json().get("result", [])
            if not results:
                return {"success": False, "message": f"❌ Ticket **{ticket_number}** not found."}

            ticket = results[0]

            def _dv(field: str) -> str:
                v = ticket.get(field, "")
                if isinstance(v, dict):
                    return str(v.get("display_value") or v.get("value") or "").strip()
                return str(v or "").strip()

            def _raw(field: str) -> str:
                v = ticket.get(field, "")
                if isinstance(v, dict):
                    return str(v.get("value") or "").strip()
                return str(v or "").strip()

            # Cache sys_id
            sys_id = _raw("sys_id") or (ticket.get("sys_id") if isinstance(ticket.get("sys_id"), str) else "")
            if sys_id:
                ServiceNowClient._sys_id_cache[ticket_number.upper()] = sys_id

            status   = TICKET_STATE_MAP.get(_raw("state"))    or _dv("state")    or "Unknown"
            priority = PRIORITY_LABEL_MAP.get(_raw("priority")) or _dv("priority") or "Unknown"
            updated  = self._format_dt(_dv("sys_updated_on"))

            # Strip "Reported by:" prefix written at creation time
            description = re.sub(r'^Reported by:.*?\n\n', '', _dv("description"), flags=re.DOTALL).strip()

            return {
                "success": True,
                "message": (
                    f"📋 **Ticket {ticket_number.upper()}**\n"
                    f"• Description:  {description[:200] + '...' if len(description) > 200 else description or 'N/A'}\n"
                    f"• Status:       {status}\n"
                    f"• Priority:     {priority}\n"
                    f"• Last Updated: {updated}\n\n"
                    f"Is there anything else you'd like to do with this ticket?"
                ),
            }
        except Exception as exc:
            return {"success": False, "message": f"❌ Could not fetch ticket {ticket_number}: {exc}"}

    def get_latest_comment(self, ticket_number: str) -> Dict:
        """Return only the latest comment/work note — no full ticket details."""
        try:
            sys_id = self._get_sys_id(ticket_number)
            if not sys_id:
                return {"success": False, "message": f"❌ Ticket **{ticket_number}** not found."}

            j_resp = ServiceNowClient._session.get(
                f"https://{SNOW_INSTANCE}/api/now/table/sys_journal_field",
                params={
                    "sysparm_query":         f"element_id={sys_id}^ORDERBYDESCsys_created_on",
                    "sysparm_fields":        "value,sys_created_on,element",
                    "sysparm_limit":         5,
                    "sysparm_display_value": "true",
                },
                timeout=SNOW_TIMEOUT,
            )
            j_resp.raise_for_status()
            entries = [
                e for e in j_resp.json().get("result", [])
                if not str(e.get("value", "")).startswith("Reported by:")
            ]

            if not entries:
                return {"success": True, "message": f"No comments found on ticket **{ticket_number}**."}

            e = entries[0]
            return {
                "success": True,
                "message": (
                    f"Latest comment on **{ticket_number.upper()}**\n"
                    f"• Date & Time: {self._format_dt(str(e.get('sys_created_on', '')))}\n"
                    f"• Note: {str(e.get('value', '')).strip()[:500]}"
                ),
            }
        except Exception as exc:
            return {"success": False, "message": f"❌ Could not fetch comments for {ticket_number}: {exc}"}

    def update_ticket(self, ticket_number: str, update_field: str, update_value: str) -> Dict:
        try:
            sys_id = self._get_sys_id(ticket_number)
            if not sys_id:
                return {"success": False, "message": f"❌ Ticket **{ticket_number}** not found."}

            field_key = update_field.lower().strip()

            # ── Priority: must set impact+urgency atomically ──────────────────
            if field_key == "priority":
                p_key = value = update_value.strip().lower()
                if p_key == "critical":
                    p_key = "high"
                if p_key not in PRIORITY_IMPACT_URGENCY:
                    p_key = "medium"
                impact, urgency_val = PRIORITY_IMPACT_URGENCY[p_key]
                resp = ServiceNowClient._session.patch(
                    f"{self.base_url}/{sys_id}",
                    json={"impact": str(impact), "urgency": str(urgency_val)},
                    timeout=SNOW_TIMEOUT,
                )
                resp.raise_for_status()
                _DISPLAY = {"high": "High (Critical)", "medium": "Medium", "low": "Low"}
                return {
                    "success": True,
                    "message": (
                        f"✅ **{ticket_number}** priority updated to **{_DISPLAY.get(p_key, p_key.capitalize())}**.\n\n"
                        f"Is there anything else I can help you with?"
                    ),
                }

            # ── State ─────────────────────────────────────────────────────────
            if field_key == "state":
                val = STATE_NAME_MAP.get(update_value.lower().strip())
                if not val:
                    return {
                        "success": False,
                        "message": f"❌ Unknown state '{update_value}'. Valid: {', '.join(STATE_NAME_MAP)}.",
                    }
                resp = ServiceNowClient._session.patch(
                    f"{self.base_url}/{sys_id}", json={"state": val}, timeout=SNOW_TIMEOUT
                )
                resp.raise_for_status()
                return {
                    "success": True,
                    "message": f"✅ **{ticket_number}** state updated to **{update_value}**.\n\nIs there anything else I can help you with?",
                }

            # ── Comment (customer-visible) ────────────────────────────────────
            # ServiceNow REST API appends to the journal when you PATCH 'comments'
            if field_key == "comment":
                resp = ServiceNowClient._session.patch(
                    f"{self.base_url}/{sys_id}", json={"comments": update_value}, timeout=SNOW_TIMEOUT
                )
                resp.raise_for_status()
                return {
                    "success": True,
                    "message": f"✅ Comment added to **{ticket_number}**.\n\nIs there anything else I can help you with?",
                }

            # ── Work notes (internal) ─────────────────────────────────────────
            if field_key == "work_notes":
                resp = ServiceNowClient._session.patch(
                    f"{self.base_url}/{sys_id}", json={"work_notes": update_value}, timeout=SNOW_TIMEOUT
                )
                resp.raise_for_status()
                return {
                    "success": True,
                    "message": f"✅ Work note added to **{ticket_number}**.\n\nIs there anything else I can help you with?",
                }

            # ── Description ───────────────────────────────────────────────────
            if field_key == "description":
                resp = ServiceNowClient._session.patch(
                    f"{self.base_url}/{sys_id}", json={"description": update_value}, timeout=SNOW_TIMEOUT
                )
                resp.raise_for_status()
                return {
                    "success": True,
                    "message": f"✅ Description updated on **{ticket_number}**.\n\nIs there anything else I can help you with?",
                }

            return {
                "success": False,
                "message": f"❌ Unsupported field '{update_field}'. Supported: priority, state, comment, work_notes, description.",
            }

        except Exception as exc:
            return {"success": False, "message": f"❌ Failed to update ticket {ticket_number}: {exc}"}


_client: Optional[ServiceNowClient] = None


def _get_client() -> ServiceNowClient:
    global _client
    if _client is None:
        _client = ServiceNowClient()
    return _client


# ── Public functions ──────────────────────────────────────────────────────────

def create_servicenow_ticket(name: str, contact: str, description: str, priority: str = "medium") -> str:
    try:
        return _get_client().create_ticket(name, contact, description, priority).get(
            "message", "✅ Ticket created successfully."
        )
    except Exception as exc:
        return f"❌ Failed to create ticket: {exc}"


def get_servicenow_ticket_status(ticket_number: str) -> str:
    try:
        return _get_client().get_ticket_status(ticket_number).get(
            "message", f"Status retrieved for {ticket_number}."
        )
    except Exception as exc:
        return f"❌ Could not check ticket {ticket_number}: {exc}"


def get_servicenow_latest_comment(ticket_number: str) -> str:
    try:
        return _get_client().get_latest_comment(ticket_number).get(
            "message", f"Comment retrieved for {ticket_number}."
        )
    except Exception as exc:
        return f"❌ Could not fetch comments for {ticket_number}: {exc}"


def update_servicenow_ticket(ticket_number: str, update_field: str, update_value: str) -> str:
    try:
        return _get_client().update_ticket(ticket_number, update_field, update_value).get(
            "message", f"✅ Ticket {ticket_number} updated."
        )
    except Exception as exc:
        return f"❌ Failed to update ticket {ticket_number}: {exc}"