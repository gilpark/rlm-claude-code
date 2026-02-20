#!/usr/bin/env python3
"""
Migration script: CausalTrace → CausalFrame

Converts existing reasoning traces to the new frame-based structure.

Usage:
    python scripts/migrate_to_frames.py [options]

Options:
    --dry-run       Show what would be migrated without making changes
    --backup        Create backup before migration (default: True)
    --rollback      Rollback to backup
    --session DIR   Migrate specific session directory
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


def create_backup(session_dir: Path) -> Path:
    """Create a backup of the session directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = session_dir.parent / f"{session_dir.name}_backup_{timestamp}"
    shutil.copytree(session_dir, backup_dir)
    return backup_dir


def rollback_to_backup(backup_dir: Path, session_dir: Path) -> None:
    """Rollback to a backup."""
    if session_dir.exists():
        shutil.rmtree(session_dir)
    shutil.copytree(backup_dir, session_dir)


def trace_to_frame_data(trace_data: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a CausalTrace to CausalFrame format.

    This is a best-effort conversion. Some fields may not have equivalents.
    """
    from src.frame.causal_frame import FrameStatus
    from src.frame.context_slice import ContextSlice

    # Map old status to new FrameStatus
    status_map = {
        "pending": FrameStatus.CREATED,
        "running": FrameStatus.RUNNING,
        "completed": FrameStatus.COMPLETED,
        "failed": FrameStatus.INVALIDATED,
    }

    old_status = trace_data.get("status", "completed")
    new_status = status_map.get(old_status, FrameStatus.COMPLETED)

    # Create context slice from trace data
    context_slice = ContextSlice(
        files=trace_data.get("files", {}),
        memory_refs=trace_data.get("memory_refs", []),
        tool_outputs=trace_data.get("tool_outputs", {}),
        token_budget=trace_data.get("token_budget", 1000),
    )

    return {
        "frame_id": trace_data.get("id", "unknown"),
        "depth": trace_data.get("depth", 0),
        "parent_id": trace_data.get("parent_id"),
        "children": trace_data.get("children", []),
        "query": trace_data.get("query", ""),
        "context_slice": {
            "files": context_slice.files,
            "memory_refs": context_slice.memory_refs,
            "tool_outputs": context_slice.tool_outputs,
            "token_budget": context_slice.token_budget,
        },
        "evidence": trace_data.get("evidence", []),
        "conclusion": trace_data.get("conclusion"),
        "confidence": trace_data.get("confidence", 0.8),
        "invalidation_condition": trace_data.get("invalidation_condition", ""),
        "status": new_status.value,
        "branched_from": None,
        "escalation_reason": None,
        "created_at": trace_data.get("created_at", datetime.now().isoformat()),
        "completed_at": trace_data.get("completed_at"),
    }


def migrate_session(session_dir: Path, dry_run: bool = False) -> int:
    """
    Migrate a single session directory.

    Returns: Number of traces migrated
    """
    reasoning_dir = session_dir / "reasoning"
    if not reasoning_dir.exists():
        print(f"  No reasoning directory in {session_dir}")
        return 0

    migrated = 0
    for trace_file in reasoning_dir.glob("*.json"):
        try:
            with open(trace_file) as f:
                trace_data = json.load(f)

            frame_data = trace_to_frame_data(trace_data)

            if dry_run:
                print(f"  Would migrate: {trace_file.name}")
            else:
                # Write new frame file
                frame_file = reasoning_dir / f"frame_{trace_file.name}"
                with open(frame_file, "w") as f:
                    json.dump(frame_data, f, indent=2)
                print(f"  Migrated: {trace_file.name} -> {frame_file.name}")

            migrated += 1
        except Exception as e:
            print(f"  Error migrating {trace_file}: {e}")

    return migrated


def main():
    parser = argparse.ArgumentParser(description="Migrate CausalTrace to CausalFrame")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated")
    parser.add_argument("--backup", action="store_true", default=True, help="Create backup")
    parser.add_argument("--no-backup", action="store_false", dest="backup", help="Skip backup")
    parser.add_argument("--rollback", type=Path, help="Rollback to backup directory")
    parser.add_argument("--session", type=Path, help="Migrate specific session")
    args = parser.parse_args()

    # Handle rollback
    if args.rollback:
        session_dir = args.session or Path.cwd()
        print(f"Rolling back {session_dir} from {args.rollback}")
        rollback_to_backup(args.rollback, session_dir)
        print("Rollback complete")
        return

    # Find sessions to migrate
    if args.session:
        sessions = [args.session]
    else:
        base_dir = Path.home() / ".claude" / "rlm-sessions"
        if not base_dir.exists():
            print(f"No sessions found at {base_dir}")
            return
        sessions = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.endswith("_backup")]

    print(f"Found {len(sessions)} sessions to migrate")

    total_migrated = 0
    for session_dir in sessions:
        print(f"\nMigrating: {session_dir}")

        # Create backup
        if args.backup and not args.dry_run:
            backup_dir = create_backup(session_dir)
            print(f"  Backup created: {backup_dir}")

        migrated = migrate_session(session_dir, dry_run=args.dry_run)
        total_migrated += migrated

    print(f"\nTotal migrated: {total_migrated} traces")
    if args.dry_run:
        print("(dry run - no changes made)")


if __name__ == "__main__":
    main()
