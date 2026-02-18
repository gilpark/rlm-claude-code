"""
Cell Manager for RLM-Claude-Code.

Implements: Session State Consolidation Plan Phase 2

Manages replayable computation units (cells) with dependency tracking
and topological execution ordering.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .session_schema import (
    Cell,
    CellIndex,
    CellIndexEntry,
    CellInput,
    CellMetadata,
    CellOutput,
    CellTrace,
    CellType,
)

if TYPE_CHECKING:
    from .session_manager import SessionManager


class CellCycleError(Exception):
    """Raised when a cell dependency would create a cycle."""

    def __init__(self, cell_id: str, dependency_id: str):
        self.cell_id = cell_id
        self.dependency_id = dependency_id
        super().__init__(f"Adding dependency {dependency_id} to {cell_id} would create a cycle")


class CellNotFoundError(Exception):
    """Raised when a cell is not found."""

    def __init__(self, cell_id: str):
        self.cell_id = cell_id
        super().__init__(f"Cell not found: {cell_id}")


class CellDependencyError(Exception):
    """Raised when a cell dependency is invalid."""

    def __init__(self, cell_id: str, dependency_id: str, reason: str):
        self.cell_id = cell_id
        self.dependency_id = dependency_id
        self.reason = reason
        super().__init__(f"Invalid dependency {dependency_id} for {cell_id}: {reason}")


class CellManager:
    """
    Manages cells with dependency resolution and topological ordering.

    Implements: Session State Consolidation Plan Phase 2

    Cells are replayable computation units that track:
    - Input (source, operation, args)
    - Output (result, error, execution time)
    - Dependencies (cells this depends on)
    - Metadata (depth, model, tokens)

    The CellManager maintains a DAG (Directed Acyclic Graph) of cells
    and can compute topological execution order.
    """

    def __init__(
        self,
        cells_dir: Path,
        session_id: str,
        session_manager: SessionManager | None = None,
    ):
        """
        Initialize CellManager.

        Args:
            cells_dir: Directory for cell storage
            session_id: Session ID for this cell manager
            session_manager: Optional SessionManager for updates
        """
        self.cells_dir = cells_dir
        self.session_id = session_id
        self.session_manager = session_manager

        # Ensure cells directory exists
        self.cells_dir.mkdir(parents=True, exist_ok=True)

        # Load or create cell index
        self._index_path = self.cells_dir / "index.json"
        self._index: CellIndex = self._load_or_create_index()

        # Cache for loaded cells
        self._cell_cache: dict[str, Cell] = {}

    def _load_or_create_index(self) -> CellIndex:
        """Load existing index or create new one."""
        if self._index_path.exists():
            with open(self._index_path) as f:
                data = json.load(f)
            return CellIndex(**data)
        else:
            return CellIndex(session_id=self.session_id)

    def _save_index(self) -> None:
        """Save cell index to disk using atomic write."""
        # Write to temp file first
        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix="index_",
            dir=self.cells_dir,
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(self._index.model_dump_json(indent=2))
            # Atomic rename
            os.rename(temp_path, self._index_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    @staticmethod
    def generate_cell_id() -> str:
        """
        Generate a unique cell ID with pattern cell_{hex8}.

        Returns:
            Cell ID string like "cell_a1b2c3d4"
        """
        hex_str = secrets.token_hex(4)  # 8 hex characters
        return f"cell_{hex_str}"

    def create_cell(
        self,
        cell_type: CellType,
        input_source: str,
        operation: str,
        args: dict[str, Any] | None = None,
        dependencies: list[str] | None = None,
        context_snapshot: dict[str, Any] | None = None,
        depth: int = 0,
        model: str | None = None,
        trace: CellTrace | None = None,
    ) -> Cell:
        """
        Create a new cell.

        Args:
            cell_type: Type of cell (REPL, TOOL, LLM_CALL, etc.)
            input_source: Source of the input ("repl", "tool", "llm_call")
            operation: Operation being performed
            args: Arguments for the operation
            dependencies: List of cell IDs this depends on
            context_snapshot: Optional context snapshot
            depth: Recursion depth
            model: Model used (for LLM calls)
            trace: Optional reasoning trace

        Returns:
            Newly created Cell

        Raises:
            CellCycleError: If dependency would create a cycle
            CellNotFoundError: If dependency doesn't exist
        """
        cell_id = self.generate_cell_id()
        dependencies = dependencies or []

        # Validate dependencies
        for dep_id in dependencies:
            if dep_id not in self._index.cells:
                raise CellNotFoundError(dep_id)
            if self._would_create_cycle(cell_id, dep_id):
                raise CellCycleError(cell_id, dep_id)

        now = time.time()
        cell = Cell(
            cell_id=cell_id,
            created_at=now,
            type=cell_type,
            input=CellInput(
                source=input_source,
                operation=operation,
                args=args or {},
                context_snapshot=context_snapshot,
            ),
            output=CellOutput(),
            dependencies=dependencies,
            dependents=[],  # Will be updated below
            metadata=CellMetadata(
                depth=depth,
                model=model,
            ),
            trace=trace,
        )

        # Update dependents of dependencies
        for dep_id in dependencies:
            dep_entry = self._index.cells[dep_id]
            if cell_id not in dep_entry.dependents:
                dep_entry.dependents.append(cell_id)

        # Add to index
        self._index.cells[cell_id] = CellIndexEntry(
            type=cell_type,
            dependencies=dependencies,
            dependents=[],
        )

        # Update roots/leaves
        if not dependencies:
            if cell_id not in self._index.roots:
                self._index.roots.append(cell_id)
        self._index.leaves.append(cell_id)

        # Remove from leaves if something depends on us
        # (already handled since dependents are updated above, but we're new)

        # Recompute execution order
        self._recompute_execution_order()

        # Update cell count in session
        if self.session_manager and self.session_manager.current_session:
            self.session_manager.current_session.cells.count = len(self._index.cells)

        # Save cell to disk
        self._save_cell(cell)

        # Save index
        self._save_index()

        # Cache the cell
        self._cell_cache[cell_id] = cell

        return cell

    def complete_cell(
        self,
        cell_id: str,
        result: Any = None,
        error: str | None = None,
        execution_time_ms: float = 0.0,
        tokens_used: int = 0,
        cached_response: str | None = None,
    ) -> Cell:
        """
        Mark a cell as completed with result or error.

        Args:
            cell_id: Cell to complete
            result: Result value
            error: Error message if failed
            execution_time_ms: Execution time in milliseconds
            tokens_used: Tokens used (for LLM calls)
            cached_response: Cached response ID if result was cached

        Returns:
            Updated Cell

        Raises:
            CellNotFoundError: If cell doesn't exist
        """
        cell = self.get_cell(cell_id)

        cell.output = CellOutput(
            result=result,
            error=error,
            execution_time_ms=execution_time_ms,
            cached_response=cached_response,
        )
        cell.metadata.tokens_used = tokens_used

        # Save updated cell
        self._save_cell(cell)

        # Update cache
        self._cell_cache[cell_id] = cell

        return cell

    def _save_cell(self, cell: Cell) -> None:
        """Save cell to disk using atomic write."""
        cell_path = self.cells_dir / f"{cell.cell_id}.json"

        fd, temp_path = tempfile.mkstemp(
            suffix=".tmp",
            prefix=f"{cell.cell_id}_",
            dir=self.cells_dir,
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(cell.model_dump_json(indent=2))
            os.rename(temp_path, cell_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def get_cell(self, cell_id: str) -> Cell:
        """
        Load a cell from disk or cache.

        Args:
            cell_id: Cell to load

        Returns:
            Cell object

        Raises:
            CellNotFoundError: If cell doesn't exist
        """
        # Check cache first
        if cell_id in self._cell_cache:
            return self._cell_cache[cell_id]

        # Check if cell exists in index
        if cell_id not in self._index.cells:
            raise CellNotFoundError(cell_id)

        # Load from disk
        cell_path = self.cells_dir / f"{cell_id}.json"
        if not cell_path.exists():
            raise CellNotFoundError(cell_id)

        with open(cell_path) as f:
            data = json.load(f)

        cell = Cell(**data)
        self._cell_cache[cell_id] = cell
        return cell

    def _would_create_cycle(self, cell_id: str, dependency_id: str) -> bool:
        """
        Check if adding dependency would create a cycle.

        Uses BFS to check if dependency_id is reachable from cell_id
        through existing dependency edges.

        Args:
            cell_id: Cell that would have the dependency
            dependency_id: Cell that would be depended on

        Returns:
            True if adding this dependency would create a cycle
        """
        # If dependency doesn't exist, no cycle possible
        if dependency_id not in self._index.cells:
            return False

        # Self-dependency
        if cell_id == dependency_id:
            return True

        # BFS from dependency_id to see if we can reach cell_id
        visited = set()
        queue = [dependency_id]

        while queue:
            current = queue.pop(0)
            if current == cell_id:
                return True
            if current in visited:
                continue
            visited.add(current)

            # Add dependencies of current cell
            if current in self._index.cells:
                for dep in self._index.cells[current].dependencies:
                    if dep not in visited:
                        queue.append(dep)

        return False

    def validate_dependencies(self, cell_id: str) -> list[str]:
        """
        Validate dependencies of a cell.

        Args:
            cell_id: Cell to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if cell_id not in self._index.cells:
            errors.append(f"Cell {cell_id} not found in index")
            return errors

        entry = self._index.cells[cell_id]

        # Check for self-dependency
        if cell_id in entry.dependencies:
            errors.append(f"Cell {cell_id} has self-dependency")

        # Check for missing dependencies
        for dep_id in entry.dependencies:
            if dep_id not in self._index.cells:
                errors.append(f"Dependency {dep_id} not found")

        # Check for cycles
        for dep_id in entry.dependencies:
            if self._would_create_cycle(cell_id, dep_id):
                errors.append(f"Dependency {dep_id} creates a cycle")

        return errors

    def _recompute_execution_order(self) -> None:
        """
        Recompute topological execution order.

        Uses Kahn's algorithm for topological sort.
        Updates self._index.execution_order and self._index.cycles_detected.
        """
        # Build adjacency list and in-degree count
        in_degree: dict[str, int] = defaultdict(int)
        adjacency: dict[str, list[str]] = defaultdict(list)

        for cell_id, entry in self._index.cells.items():
            in_degree[cell_id]  # Initialize
            for dep_id in entry.dependencies:
                adjacency[dep_id].append(cell_id)
                in_degree[cell_id] += 1

        # Start with cells that have no dependencies
        queue = [cid for cid, deg in in_degree.items() if deg == 0]
        queue.sort()  # Deterministic ordering

        execution_order: list[str] = []
        cycles_detected: list[str] = []

        while queue:
            current = queue.pop(0)
            execution_order.append(current)

            for dependent in adjacency[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()

        # Any cells not in execution order are part of a cycle
        for cell_id in self._index.cells:
            if cell_id not in execution_order:
                cycles_detected.append(cell_id)

        self._index.execution_order = execution_order
        self._index.cycles_detected = cycles_detected if cycles_detected else None

        # Update roots and leaves
        self._index.roots = [
            cid for cid in execution_order
            if not self._index.cells[cid].dependencies
        ]
        self._index.leaves = [
            cid for cid in execution_order
            if not self._index.cells[cid].dependents
        ]

    def get_execution_order(self) -> list[str]:
        """
        Get topological execution order of all cells.

        Returns:
            List of cell IDs in execution order
        """
        return self._index.execution_order.copy()

    def get_dependency_chain(self, cell_id: str) -> list[str]:
        """
        Get transitive dependencies of a cell in execution order.

        Args:
            cell_id: Cell to get dependencies for

        Returns:
            List of cell IDs that must execute before this cell
        """
        if cell_id not in self._index.cells:
            return []

        # BFS to collect all transitive dependencies
        visited = set()
        queue = list(self._index.cells[cell_id].dependencies)

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current in self._index.cells:
                for dep in self._index.cells[current].dependencies:
                    if dep not in visited:
                        queue.append(dep)

        # Sort by execution order
        return [cid for cid in self._index.execution_order if cid in visited]

    def get_dependents(self, cell_id: str) -> list[str]:
        """
        Get all cells that depend on a given cell.

        Args:
            cell_id: Cell to get dependents for

        Returns:
            List of cell IDs that depend on this cell
        """
        if cell_id not in self._index.cells:
            return []

        return self._index.cells[cell_id].dependents.copy()

    def invalidate_cell(self, cell_id: str) -> list[str]:
        """
        Invalidate a cell and all cells that depend on it.

        This marks cells as needing re-execution when a file changes.

        Args:
            cell_id: Cell to invalidate

        Returns:
            List of all invalidated cell IDs
        """
        if cell_id not in self._index.cells:
            return []

        invalidated = [cell_id]

        # BFS to find all dependents
        queue = list(self._index.cells[cell_id].dependents)

        while queue:
            current = queue.pop(0)
            if current in invalidated:
                continue
            invalidated.append(current)

            if current in self._index.cells:
                queue.extend(self._index.cells[current].dependents)

        return invalidated

    def replay_cell(self, cell_id: str) -> Any:
        """
        Replay a cell's computation.

        This is a placeholder for the actual replay logic, which would
        need to integrate with the REPL environment.

        Args:
            cell_id: Cell to replay

        Returns:
            Cell result

        Raises:
            CellNotFoundError: If cell doesn't exist
            NotImplementedError: Always (needs REPL integration)
        """
        cell = self.get_cell(cell_id)

        # TODO: Integrate with RLMEnvironment to actually replay
        # For now, return the cached result
        if cell.output.error:
            raise RuntimeError(f"Cell {cell_id} had error: {cell.output.error}")

        return cell.output.result

    def get_root_cells(self) -> list[str]:
        """Get cells with no dependencies."""
        return self._index.roots.copy()

    def get_leaf_cells(self) -> list[str]:
        """Get cells with no dependents."""
        return self._index.leaves.copy()

    def get_cells_by_type(self, cell_type: CellType) -> list[str]:
        """
        Get all cells of a specific type.

        Args:
            cell_type: Type to filter by

        Returns:
            List of cell IDs of the given type
        """
        return [
            cid for cid, entry in self._index.cells.items()
            if entry.type == cell_type
        ]

    def get_cell_count(self) -> int:
        """Get total number of cells."""
        return len(self._index.cells)

    def has_cycles(self) -> bool:
        """Check if there are any cycles in the cell graph."""
        return self._index.cycles_detected is not None and len(self._index.cycles_detected) > 0

    def get_cycles(self) -> list[str]:
        """Get cells that are part of cycles."""
        return self._index.cycles_detected or []

    def clear_cache(self) -> None:
        """Clear the cell cache."""
        self._cell_cache.clear()

    def reload_index(self) -> None:
        """Reload the cell index from disk."""
        self._index = self._load_or_create_index()
        self._cell_cache.clear()


__all__ = [
    "CellManager",
    "CellCycleError",
    "CellNotFoundError",
    "CellDependencyError",
]
