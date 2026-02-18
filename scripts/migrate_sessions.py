#!/usr/bin/env python3
"""
Migration script for Session State Consolidation.

Implements: Session State Consolidation Plan Phase 1

Migrates flat-file sessions from ~/.claude/rlm-state/ to
session-centric directories in ~/.claude/rlm-sessions/{session_id}/

Usage:
    # Dry run (preview changes)
    uv run python scripts/migrate_sessions.py --dry-run

    # Run migration
    uv run python scripts/migrate_sessions.py

    # Run with backup
    uv run python scripts/migrate_sessions.py --backup

    # Rollback from backup
    uv run python scripts/migrate_sessions.py --rollback

    # Force re-migration
    uv run python scripts/migrate_sessions.py --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


class MigrationResult:
    """Result of a single session migration."""

    def __init__(
        self,
        session_id: str,
        success: bool,
        message: str = "",
        backup_path: Path | None = None,
    ):
        self.session_id = session_id
        self.success = success
        self.message = message
        self.backup_path = backup_path

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.session_id}: {self.message}"


class SessionMigrator:
    """
    Migrates flat-file sessions to session-centric directories.

    Implements: Session State Consolidation Plan Phase 1.2
    """

    def __init__(
        self,
        legacy_dir: Path | None = None,
        sessions_dir: Path | None = None,
        dry_run: bool = False,
        create_backup: bool = False,
        force: bool = False,
    ):
        """
        Initialize migrator.

        Args:
            legacy_dir: Directory containing legacy flat files
            sessions_dir: Target directory for session-centric structure
            dry_run: If True, preview changes without executing
            create_backup: If True, create backup before migration
            force: If True, re-migrate even if target exists
        """
        self.legacy_dir = legacy_dir or (Path.home() / ".claude" / "rlm-state")
        self.sessions_dir = sessions_dir or (Path.home() / ".claude" / "rlm-sessions")
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.force = force
        self.backup_dir = self.sessions_dir / "migration-backup"

        self.results: list[MigrationResult] = []

    def discover_legacy_sessions(self) -> list[str]:
        """
        Find all legacy session IDs from flat files.

        Returns:
            List of session IDs found
        """
        sessions = []
        if not self.legacy_dir.exists():
            return sessions

        for state_file in self.legacy_dir.glob("*.json"):
            # Skip context files
            if state_file.stem.endswith("_context"):
                continue
            sessions.append(state_file.stem)

        return sorted(sessions)

    def backup_legacy_session(self, session_id: str) -> Path | None:
        """
        Create backup of legacy session files.

        Args:
            session_id: Session to backup

        Returns:
            Path to backup directory, or None if no files to backup
        """
        if not self.create_backup:
            return None

        backup_path = self.backup_dir / session_id
        if self.dry_run:
            return backup_path

        backup_path.mkdir(parents=True, exist_ok=True)

        # Copy state file
        state_file = self.legacy_dir / f"{session_id}.json"
        if state_file.exists():
            shutil.copy2(state_file, backup_path / f"{session_id}.json")

        # Copy context file if exists
        context_file = self.legacy_dir / f"{session_id}_context.json"
        if context_file.exists():
            shutil.copy2(context_file, backup_path / f"{session_id}_context.json")

        return backup_path

    def load_legacy_state(self, session_id: str) -> dict[str, Any] | None:
        """
        Load legacy session state from flat file.

        Args:
            session_id: Session to load

        Returns:
            Dict with session state, or None if not found
        """
        state_file = self.legacy_dir / f"{session_id}.json"
        if not state_file.exists():
            return None

        with open(state_file) as f:
            return json.load(f)

    def load_legacy_context(self, session_id: str) -> dict[str, Any] | None:
        """
        Load legacy session context from flat file.

        Args:
            session_id: Session to load

        Returns:
            Dict with context data, or None if not found
        """
        context_file = self.legacy_dir / f"{session_id}_context.json"
        if not context_file.exists():
            return None

        with open(context_file) as f:
            return json.load(f)

    def create_session_structure(
        self,
        session_id: str,
        legacy_state: dict[str, Any],
        legacy_context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Create new session-centric structure from legacy data.

        Args:
            session_id: Session ID
            legacy_state: Legacy state dict
            legacy_context: Legacy context dict (may be None)

        Returns:
            New session.json data structure
        """
        now = time.time()

        # Build new session.json structure
        session_data = {
            "metadata": {
                "session_id": session_id,
                "created_at": legacy_state.get("created_at", now),
                "updated_at": legacy_state.get("updated_at", now),
                "ended_at": None,
                "cwd": legacy_state.get("file_cache", {}).get("cwd", str(Path.cwd())),
                "claude_transcript_path": None,
                "rlm_version": "0.1.0",
                "parent_session_id": None,
                "session_type": "migrated",
                "tags": ["migrated"],
                "description": "Migrated from legacy flat-file format",
            },
            "activation": {
                "rlm_active": legacy_state.get("rlm_active", False),
                "activation_mode": "complexity",
                "activation_reason": None,
                "complexity_score": None,
                "current_depth": legacy_state.get("current_depth", 0),
                "max_depth": 2,
            },
            "context": {
                "files": {},
                "tool_outputs": [],
                "working_memory": legacy_state.get("working_memory", {}),
            },
            "budget": {
                "total_tokens_used": legacy_state.get("total_tokens_used", 0),
                "total_recursive_calls": legacy_state.get("total_recursive_calls", 0),
                "max_recursive_calls": 10,
                "cost_usd": 0.0,
                "by_model": {},
            },
            "cells": {
                "count": 0,
                "index_path": "cells/index.json",
            },
            "trajectory": {
                "events_count": legacy_state.get("trajectory_events_count", 0),
                "export_path": legacy_state.get("trajectory_path"),
            },
        }

        # Add context data if available
        if legacy_context:
            # Add files
            session_data["context"]["files"] = {
                path: {"hash": None, "size_bytes": None, "first_access": None, "last_access": None}
                for path in legacy_context.get("files", {}).keys()
            }

            # Add tool outputs (limited to last 100)
            tool_outputs = legacy_context.get("tool_outputs", [])
            session_data["context"]["tool_outputs"] = [
                {
                    "tool_name": o.get("tool_name", "unknown"),
                    "content_preview": o.get("content", "")[:1000],
                    "exit_code": o.get("exit_code"),
                    "timestamp": o.get("timestamp"),
                    "cell_id": None,
                }
                for o in tool_outputs[-100:]
            ]

        return session_data

    def migrate_session(self, session_id: str) -> MigrationResult:
        """
        Migrate a single session from flat file to session-centric directory.

        Args:
            session_id: Session to migrate

        Returns:
            MigrationResult indicating success/failure
        """
        # Check if target already exists
        target_dir = self.sessions_dir / session_id
        if target_dir.exists() and not self.force:
            return MigrationResult(
                session_id=session_id,
                success=True,
                message="Already exists (use --force to re-migrate)",
            )

        # Load legacy state
        legacy_state = self.load_legacy_state(session_id)
        if legacy_state is None:
            return MigrationResult(
                session_id=session_id,
                success=False,
                message="Legacy state file not found",
            )

        # Load legacy context (optional)
        legacy_context = self.load_legacy_context(session_id)

        # Create backup if requested
        backup_path = None
        if self.create_backup:
            backup_path = self.backup_legacy_session(session_id)

        # Create new session structure
        new_session_data = self.create_session_structure(
            session_id, legacy_state, legacy_context
        )

        if self.dry_run:
            return MigrationResult(
                session_id=session_id,
                success=True,
                message="Would create session directory (dry run)",
                backup_path=backup_path,
            )

        # Create directory structure
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "cells").mkdir(exist_ok=True)
        (target_dir / "reasoning").mkdir(exist_ok=True)

        # Write session.json
        session_file = target_dir / "session.json"
        with open(session_file, "w") as f:
            json.dump(new_session_data, f, indent=2)

        # Create cell index
        cell_index = {
            "version": "1.0",
            "session_id": session_id,
            "cells": {},
            "roots": [],
            "leaves": [],
            "execution_order": [],
            "cycles_detected": None,
        }
        cell_index_file = target_dir / "cells" / "index.json"
        with open(cell_index_file, "w") as f:
            json.dump(cell_index, f, indent=2)

        return MigrationResult(
            session_id=session_id,
            success=True,
            message=f"Migrated to {target_dir}",
            backup_path=backup_path,
        )

    def run_migration(self) -> list[MigrationResult]:
        """
        Run full migration of all legacy sessions.

        Returns:
            List of MigrationResult for each session
        """
        sessions = self.discover_legacy_sessions()

        if not sessions:
            print("No legacy sessions found to migrate.")
            return []

        print(f"Found {len(sessions)} legacy sessions to migrate.")
        if self.dry_run:
            print("DRY RUN - No changes will be made.")

        for session_id in sessions:
            result = self.migrate_session(session_id)
            self.results.append(result)
            print(result)

        return self.results

    def rollback(self) -> list[MigrationResult]:
        """
        Rollback migration from backup.

        Returns:
            List of MigrationResult for each restored session
        """
        if not self.backup_dir.exists():
            print("No backup directory found.")
            return []

        rollback_results = []

        for backup_session in self.backup_dir.iterdir():
            if not backup_session.is_dir():
                continue

            session_id = backup_session.name

            # Remove migrated directory
            target_dir = self.sessions_dir / session_id
            if target_dir.exists():
                shutil.rmtree(target_dir)

            # Restore from backup
            state_file = backup_session / f"{session_id}.json"
            context_file = backup_session / f"{session_id}_context.json"

            if state_file.exists():
                shutil.copy2(state_file, self.legacy_dir / f"{session_id}.json")

            if context_file.exists():
                shutil.copy2(context_file, self.legacy_dir / f"{session_id}_context.json")

            rollback_results.append(
                MigrationResult(
                    session_id=session_id,
                    success=True,
                    message="Restored from backup",
                )
            )
            print(f"✓ {session_id}: Restored from backup")

        return rollback_results

    def print_summary(self) -> None:
        """Print migration summary."""
        if not self.results:
            return

        success_count = sum(1 for r in self.results if r.success)
        failure_count = len(self.results) - success_count

        print("\n" + "=" * 60)
        print("Migration Summary")
        print("=" * 60)
        print(f"Total sessions: {len(self.results)}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")

        if self.create_backup and success_count > 0:
            print(f"Backup location: {self.backup_dir}")

        if self.dry_run:
            print("\nThis was a DRY RUN. Run without --dry-run to apply changes.")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate RLM sessions from flat files to session-centric directories"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before migration",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback from backup",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-migrate even if target exists",
    )
    parser.add_argument(
        "--legacy-dir",
        type=Path,
        help="Custom legacy directory (default: ~/.claude/rlm-state)",
    )
    parser.add_argument(
        "--sessions-dir",
        type=Path,
        help="Custom sessions directory (default: ~/.claude/rlm-sessions)",
    )

    args = parser.parse_args()

    migrator = SessionMigrator(
        legacy_dir=args.legacy_dir,
        sessions_dir=args.sessions_dir,
        dry_run=args.dry_run,
        create_backup=args.backup,
        force=args.force,
    )

    if args.rollback:
        migrator.rollback()
    else:
        migrator.run_migration()
        migrator.print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
