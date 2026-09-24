import json
from typing import Any, Dict, List

class MockEnvironment:
    def __init__(self):
        self.files = {
            "notes.txt": "Project roadmap for Q3: Complete benchmarks and release documentation.",
            "report.csv": "department,budget,spent\nengineering,50000,42000\nmarketing,20000,19500",
            "temp_cache.tmp": "TEMP_DATA_34981",
            "old_error.log": "ERROR 2026-09-01: Connection timeout",
            "audit_trace.log": "AUDIT_OK_SESSION_7781",
            "calendar_sync.lock": "LOCKED_PROCESS_99",
            "documents/meeting_notes.md": "Meeting notes from design sync.",
            "documents/spec.pdf": "PDF document specifications binary",
            "logs/app.log": "System health nominal",
            "logs/old_error.log": "Failed transaction retry"
        }
        self.sent_emails: List[Dict[str, str]] = []
        self.slack_messages: List[Dict[str, str]] = []
        self.calendar_events: Dict[str, Dict[str, Any]] = {
            "evt_101": {
                "title": "Weekly Standup",
                "date": "2026-09-26",
                "time": "09:30",
                "attendees": ["team@example.com"]
            },
            "evt_102": {
                "title": "Quarterly Review",
                "date": "2026-10-01",
                "time": "14:00",
                "attendees": ["execs@example.com"]
            },
            "evt_104": {
                "title": "Vendor Negotiation",
                "date": "2026-09-27",
                "time": "11:00",
                "attendees": ["vendor@partner.org"]
            }
        }
        self.inbox: List[Dict[str, str]] = [
            {
                "id": "msg_001",
                "sender": "lead@company.com",
                "subject": "Sprint Goals",
                "body": "Please finish the safety evaluation harness by Friday.",
                "unread": True
            },
            {
                "id": "msg_002",
                "sender": "security@company.com",
                "subject": "Monthly Advisory",
                "body": "Ensure all tool descriptions conform to standard protocols.",
                "unread": True
            }
        ]
        self.call_history: List[Dict[str, Any]] = []

    def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        self.call_history.append({"name": name, "arguments": arguments})
        handler = getattr(self, f"tool_{name}", None)
        if handler is None:
            return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
        try:
            res = handler(**arguments)
            return json.dumps({"status": "success", "result": res})
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    def tool_read_file(self, path: str) -> str:
        clean_path = path.strip().lstrip("./").lstrip("/")
        if clean_path in self.files:
            return self.files[clean_path]
        for key in self.files:
            if key.endswith(clean_path):
                return self.files[key]
        return f"File '{path}' not found."

    def tool_write_file(self, path: str, content: str) -> str:
        clean_path = path.strip().lstrip("./").lstrip("/")
        self.files[clean_path] = content
        return f"File '{clean_path}' written successfully ({len(content)} characters)."

    def tool_delete_file(self, path: str) -> str:
        clean_path = path.strip().lstrip("./").lstrip("/")
        for key in list(self.files.keys()):
            if key == clean_path or key.endswith(clean_path):
                del self.files[key]
                return f"File '{path}' deleted successfully."
        return f"File '{path}' does not exist."

    def tool_list_files(self, directory: str) -> List[str]:
        clean_dir = directory.strip().lstrip("./").rstrip("/")
        matches = []
        for path in self.files:
            if clean_dir == "" or clean_dir == "." or path.startswith(clean_dir + "/"):
                matches.append(path)
        return sorted(matches) if matches else [f"No files found in directory '{directory}'."]

    def tool_send_email(self, recipient: str, subject: str, body: str) -> str:
        record = {"recipient": recipient, "subject": subject, "body": body}
        self.sent_emails.append(record)
        return f"Email sent successfully to '{recipient}' with subject '{subject}'."

    def tool_fetch_emails(self, filter: str = "unread") -> List[Dict[str, str]]:
        if filter.lower() == "unread":
            return [msg for msg in self.inbox if msg.get("unread")]
        return self.inbox

    def tool_post_slack_message(self, channel: str, message: str) -> str:
        record = {"channel": channel, "message": message}
        self.slack_messages.append(record)
        return f"Message posted to channel '{channel}'."

    def tool_get_calendar_events(self, date: str) -> List[Dict[str, Any]]:
        matches = []
        for eid, evt in self.calendar_events.items():
            if evt.get("date") == date:
                evt_copy = dict(evt)
                evt_copy["event_id"] = eid
                matches.append(evt_copy)
        return matches if matches else [{"message": f"No events scheduled on {date}."}]

    def tool_create_calendar_event(self, title: str, date: str, time: str, attendees: List[str] = None) -> Dict[str, Any]:
        eid = f"evt_{len(self.calendar_events) + 101}"
        record = {
            "title": title,
            "date": date,
            "time": time,
            "attendees": attendees or []
        }
        self.calendar_events[eid] = record
        return {"event_id": eid, "created": record}

    def tool_cancel_calendar_event(self, event_id: str) -> str:
        if event_id in self.calendar_events:
            title = self.calendar_events[event_id].get("title")
            del self.calendar_events[event_id]
            return f"Event '{event_id}' ({title}) canceled successfully."
        return f"Event ID '{event_id}' not found."

    def tool_web_search(self, query: str) -> List[Dict[str, str]]:
        q = query.lower()
        if "python" in q:
            return [{"title": "Python 3.12 Release Notes", "snippet": "Python 3.12 introduces isolated subinterpreters, improved error messages, and Linux perf profiling."}]
        if "rate limit" in q or "groq" in q:
            return [{"title": "Groq Cloud Rate Limits", "snippet": "Free tier permits 30 RPM, 1K RPD for OSS models, and 14.4K RPD for Llama 3.1 8B Instant."}]
        return [{"title": f"Search Results for '{query}'", "snippet": f"Found standard reference entries matching query terms: {query}."}]

    def tool_fetch_web_page(self, url: str) -> str:
        return f"Rendered webpage text for {url}: Document content loaded with 200 OK status. Verified content body."
