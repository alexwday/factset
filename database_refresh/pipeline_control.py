"""
Shared pipeline control flags for scheduled stage-by-stage execution.

This module stores run-state on NAS so independently scheduled stages can:
- prevent overlapping runs,
- skip downstream stages when upstream has no work or fails,
- and finalize/clear lock state in Stage 9.
"""

from __future__ import annotations

import io
import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from smb.SMBConnection import SMBConnection

DEFAULT_REFRESH_PATH = (
    "Finance Data and Analytics/DSA/Earnings Call Transcripts/Outputs/Refresh"
)
STATE_FILE_NAME = "pipeline_control_flags.json"
TERMINAL_STAGE_STATUSES = {
    "completed_successfully",
    "completed_no_files",
    "completed_no_content",
    "completed_all_processed",
    "completed_no_work",
    "skipped_overlap",
    "skipped_by_flag",
    "failed",
}

STAGE_PREREQUISITES = {
    "stage_03_extract_content": ("stage_02_database_sync", {"completed_successfully"}),
    "stage_04_validate_structure": ("stage_03_extract_content", {"completed_successfully"}),
    "stage_05_qa_pairing": ("stage_04_validate_structure", {"completed_successfully"}),
    "stage_06_llm_classification": (
        "stage_05_qa_pairing",
        {"completed_successfully", "completed_all_processed"},
    ),
    "stage_07_llm_summarization": ("stage_06_llm_classification", {"completed_successfully"}),
    "stage_08_embeddings_generation": ("stage_07_llm_summarization", {"completed_successfully"}),
}


def _utcnow() -> datetime:
    return datetime.utcnow()


def _utcnow_iso() -> str:
    return _utcnow().replace(microsecond=0).isoformat() + "Z"


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _nas_path_join(*parts: str) -> str:
    clean_parts = []
    for part in parts:
        if part:
            clean = str(part).strip("/")
            if clean:
                clean_parts.append(clean)
    return "/".join(clean_parts)


def _nas_file_exists(conn: SMBConnection, path: str) -> bool:
    try:
        conn.getAttributes(os.getenv("NAS_SHARE_NAME"), path)
        return True
    except Exception:
        return False


def _nas_create_directory_recursive(conn: SMBConnection, dir_path: str) -> bool:
    normalized_path = (dir_path or "").strip("/").rstrip("/")
    if not normalized_path:
        return False

    path_parts = [part for part in normalized_path.split("/") if part]
    current_path = ""
    for part in path_parts:
        current_path = f"{current_path}/{part}" if current_path else part
        if _nas_file_exists(conn, current_path):
            continue
        try:
            conn.createDirectory(os.getenv("NAS_SHARE_NAME"), current_path)
        except Exception:
            if not _nas_file_exists(conn, current_path):
                return False
    return True


def _nas_download_json(conn: SMBConnection, path: str) -> Optional[Dict[str, Any]]:
    try:
        file_obj = io.BytesIO()
        conn.retrieveFile(os.getenv("NAS_SHARE_NAME"), path, file_obj)
        file_obj.seek(0)
        raw = file_obj.read()
        if not raw:
            return None
        parsed = json.loads(raw.decode("utf-8"))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _nas_upload_json(conn: SMBConnection, path: str, payload: Dict[str, Any]) -> bool:
    parent_dir = "/".join(path.split("/")[:-1])
    if parent_dir and not _nas_create_directory_recursive(conn, parent_dir):
        return False

    encoded = json.dumps(payload, indent=2).encode("utf-8")
    file_obj = io.BytesIO(encoded)
    try:
        conn.storeFile(os.getenv("NAS_SHARE_NAME"), path, file_obj)
        return True
    except Exception:
        return False


def _default_state() -> Dict[str, Any]:
    return {
        "version": 1,
        "run_id": None,
        "lock_active": False,
        "lock_owner_stage": None,
        "lock_acquired_at": None,
        "lock_heartbeat_at": None,
        "skip_downstream": False,
        "skip_reason": None,
        "skip_details": {},
        "failed_stage": None,
        "failure_message": None,
        "stages": {},
        "last_updated": _utcnow_iso(),
        "last_completed_run_id": None,
        "last_final_status": None,
        "last_completed_at": None,
    }


def _refresh_path_from_config(config: Dict[str, Any]) -> str:
    return (
        config.get("stage_02_database_sync", {}).get("refresh_output_path")
        or config.get("stage_09_master_consolidation", {}).get("refresh_folder_path")
        or DEFAULT_REFRESH_PATH
    )


def _state_path_from_config(config: Dict[str, Any]) -> str:
    refresh_path = _refresh_path_from_config(config)
    return _nas_path_join(refresh_path, STATE_FILE_NAME)


def _load_state(conn: SMBConnection, config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    state_path = _state_path_from_config(config)
    state = _nas_download_json(conn, state_path) or _default_state()
    if "stages" not in state or not isinstance(state["stages"], dict):
        state["stages"] = {}
    return state, state_path


def _save_state(conn: SMBConnection, state_path: str, state: Dict[str, Any]) -> bool:
    state["last_updated"] = _utcnow_iso()
    return _nas_upload_json(conn, state_path, state)


def _is_stale_lock(state: Dict[str, Any], ttl_minutes: int) -> bool:
    if not state.get("lock_active", False):
        return False
    heartbeat = (
        _parse_iso(state.get("lock_heartbeat_at"))
        or _parse_iso(state.get("lock_acquired_at"))
    )
    if not heartbeat:
        return True
    return _utcnow() - heartbeat > timedelta(minutes=ttl_minutes)


def acquire_run_lock(
    conn: SMBConnection,
    config: Dict[str, Any],
    stage_name: str,
    ttl_minutes: int = 90,
) -> Tuple[bool, Optional[str], str]:
    state, state_path = _load_state(conn, config)
    now_iso = _utcnow_iso()

    if state.get("lock_active", False) and not _is_stale_lock(state, ttl_minutes):
        return False, state.get("run_id"), "active_lock"

    run_id = _utcnow().strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    stale_replaced = state.get("lock_active", False)

    state.update(
        {
            "run_id": run_id,
            "lock_active": True,
            "lock_owner_stage": stage_name,
            "lock_acquired_at": now_iso,
            "lock_heartbeat_at": now_iso,
            "skip_downstream": False,
            "skip_reason": None,
            "skip_details": {},
            "failed_stage": None,
            "failure_message": None,
            "stages": {},
        }
    )
    state["stages"][stage_name] = {"status": "running", "updated_at": now_iso}
    if stale_replaced:
        state["stale_lock_replaced_at"] = now_iso

    if not _save_state(conn, state_path, state):
        return False, None, "state_write_failed"
    return True, run_id, "acquired"


def mark_stage_running(
    conn: SMBConnection,
    config: Dict[str, Any],
    stage_name: str,
) -> bool:
    state, state_path = _load_state(conn, config)
    if not state.get("lock_active", False):
        return False
    now_iso = _utcnow_iso()
    state["lock_owner_stage"] = stage_name
    state["lock_heartbeat_at"] = now_iso
    state["stages"][stage_name] = {"status": "running", "updated_at": now_iso}
    return _save_state(conn, state_path, state)


def mark_stage_success(
    conn: SMBConnection,
    config: Dict[str, Any],
    stage_name: str,
    status: str = "completed_successfully",
) -> bool:
    state, state_path = _load_state(conn, config)
    now_iso = _utcnow_iso()
    state["lock_heartbeat_at"] = now_iso
    state["stages"][stage_name] = {"status": status, "updated_at": now_iso}
    return _save_state(conn, state_path, state)


def mark_skip_downstream(
    conn: SMBConnection,
    config: Dict[str, Any],
    stage_name: str,
    reason: str,
    details: Optional[Dict[str, Any]] = None,
    stage_status: str = "completed_no_work",
) -> bool:
    state, state_path = _load_state(conn, config)
    now_iso = _utcnow_iso()
    state["lock_heartbeat_at"] = now_iso
    state["skip_downstream"] = True
    state["skip_reason"] = reason
    state["skip_details"] = details or {}
    state["stages"][stage_name] = {"status": stage_status, "updated_at": now_iso}
    return _save_state(conn, state_path, state)


def mark_stage_failure(
    conn: SMBConnection,
    config: Dict[str, Any],
    stage_name: str,
    failure_message: str,
) -> bool:
    state, state_path = _load_state(conn, config)
    now_iso = _utcnow_iso()
    state["lock_heartbeat_at"] = now_iso
    state["failed_stage"] = stage_name
    state["failure_message"] = failure_message
    state["skip_downstream"] = True
    state["skip_reason"] = f"failure:{stage_name}"
    state["stages"][stage_name] = {"status": "failed", "updated_at": now_iso}
    return _save_state(conn, state_path, state)


def should_skip_stage(
    conn: SMBConnection,
    config: Dict[str, Any],
    stage_name: str,
) -> Tuple[bool, str]:
    state, _ = _load_state(conn, config)
    lock_active = state.get("lock_active", False)
    stage_status = state.get("stages", {}).get(stage_name, {}).get("status")

    if stage_name == "stage_09_master_consolidation":
        if not lock_active:
            return True, "no_active_run_lock"
        if stage_status in TERMINAL_STAGE_STATUSES:
            return True, f"already_executed:{stage_status}"
        if state.get("failed_stage") or state.get("skip_downstream"):
            return False, ""
        stage8_status = (
            state.get("stages", {})
            .get("stage_08_embeddings_generation", {})
            .get("status")
        )
        if stage8_status not in {"completed_successfully"}:
            return True, f"prerequisite_not_ready:stage_08_embeddings_generation:{stage8_status}"
        return False, ""

    # Stages 3-8 are run only when there is an active run lock and no skip/failure.
    if not lock_active:
        return True, "no_active_run_lock"
    if state.get("failed_stage"):
        return True, f"pipeline_failed:{state['failed_stage']}"
    if state.get("skip_downstream"):
        return True, f"skip_downstream:{state.get('skip_reason', 'unknown')}"
    if stage_status in TERMINAL_STAGE_STATUSES:
        return True, f"already_executed:{stage_status}"

    prerequisite = STAGE_PREREQUISITES.get(stage_name)
    if prerequisite:
        prereq_stage, allowed_statuses = prerequisite
        prereq_status = state.get("stages", {}).get(prereq_stage, {}).get("status")
        if prereq_status not in allowed_statuses:
            return True, f"prerequisite_not_ready:{prereq_stage}:{prereq_status}"
    return False, ""


def finalize_run(
    conn: SMBConnection,
    config: Dict[str, Any],
    stage_name: str,
    final_status: str,
) -> bool:
    state, state_path = _load_state(conn, config)
    now_iso = _utcnow_iso()

    state["stages"][stage_name] = {"status": final_status, "updated_at": now_iso}
    state["last_completed_run_id"] = state.get("run_id")
    state["last_final_status"] = final_status
    state["last_completed_at"] = now_iso

    # Clear active run state for next schedule window.
    state["run_id"] = None
    state["lock_active"] = False
    state["lock_owner_stage"] = None
    state["lock_acquired_at"] = None
    state["lock_heartbeat_at"] = None
    state["skip_downstream"] = False
    state["skip_reason"] = None
    state["skip_details"] = {}
    state["failed_stage"] = None
    state["failure_message"] = None

    return _save_state(conn, state_path, state)
