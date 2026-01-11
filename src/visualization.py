"""
Visualization for trajectory viewing and replay.

Implements: Spec §8.1 Phase 4 - Visualization
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .trajectory import TrajectoryEvent, TrajectoryEventType


class ExportFormat(Enum):
    """Supported export formats."""

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class TimelineEntry:
    """Entry in the visualization timeline."""

    timestamp: float
    event_type: TrajectoryEventType
    label: str
    content: str
    depth: int
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def formatted_time(self) -> str:
        """Format timestamp for display."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S.%f")[:-3]

    @property
    def indent(self) -> str:
        """Indentation based on depth."""
        return "  " * self.depth


@dataclass
class TrajectoryStats:
    """Statistics about a trajectory."""

    total_events: int
    total_duration_ms: float
    max_depth: int
    events_by_type: dict[str, int]
    tokens_used: int
    repl_executions: int
    recursive_calls: int


class TrajectoryVisualizer:
    """
    Visualize trajectories for debugging and analysis.

    Implements: Spec §8.1 Phase 4 - Trajectory viewer
    """

    # Icons for event types
    ICONS: dict[TrajectoryEventType, str] = {
        TrajectoryEventType.RLM_START: "🚀",
        TrajectoryEventType.ANALYZE: "🔍",
        TrajectoryEventType.REPL_EXEC: "⚡",
        TrajectoryEventType.REPL_RESULT: "📤",
        TrajectoryEventType.REASON: "💭",
        TrajectoryEventType.RECURSE_START: "🔄",
        TrajectoryEventType.RECURSE_END: "↩️",
        TrajectoryEventType.FINAL: "🏁",
        TrajectoryEventType.ERROR: "❌",
        TrajectoryEventType.TOOL_USE: "🔧",
    }

    def __init__(self, events: list[TrajectoryEvent] | None = None):
        """
        Initialize visualizer.

        Args:
            events: Optional list of events to visualize
        """
        self.events = events or []
        self._timeline: list[TimelineEntry] | None = None

    def load_from_file(self, path: Path) -> None:
        """
        Load trajectory from file.

        Args:
            path: Path to trajectory file (JSON or JSONL)
        """
        content = path.read_text()
        self.events = []

        if path.suffix == ".jsonl":
            for line in content.strip().split("\n"):
                if line:
                    data = json.loads(line)
                    self.events.append(self._event_from_dict(data))
        else:
            data = json.loads(content)
            if isinstance(data, list):
                self.events = [self._event_from_dict(e) for e in data]
            else:
                self.events = [self._event_from_dict(data)]

        self._timeline = None

    def _event_from_dict(self, data: dict[str, Any]) -> TrajectoryEvent:
        """Convert dict to TrajectoryEvent."""
        # Handle both "type" and "event_type" keys for compatibility
        event_type_value = data.get("type") or data.get("event_type")
        return TrajectoryEvent(
            type=TrajectoryEventType(event_type_value),
            depth=data.get("depth", 0),
            content=data.get("content", ""),
            metadata=data.get("metadata"),
            timestamp=data.get("timestamp", 0.0),
        )

    def build_timeline(self) -> list[TimelineEntry]:
        """
        Build timeline from events.

        Returns:
            List of timeline entries
        """
        if self._timeline is not None:
            return self._timeline

        timeline: list[TimelineEntry] = []
        start_times: dict[str, float] = {}

        for event in self.events:
            # Track start times for duration calculation
            key = f"{event.type.value}:{event.depth}"
            if event.type in (
                TrajectoryEventType.RLM_START,
                TrajectoryEventType.REPL_EXEC,
                TrajectoryEventType.RECURSE_START,
            ):
                start_times[key] = event.timestamp

            # Calculate duration for end events
            duration_ms = None
            if event.type == TrajectoryEventType.FINAL:
                start_key = f"{TrajectoryEventType.RLM_START.value}:{event.depth}"
                if start_key in start_times:
                    duration_ms = (event.timestamp - start_times[start_key]) * 1000

            # Create timeline entry
            entry = TimelineEntry(
                timestamp=event.timestamp,
                event_type=event.type,
                label=self._get_label(event),
                content=self._format_content(event),
                depth=event.depth,
                duration_ms=duration_ms,
                metadata=event.metadata or {},
            )
            timeline.append(entry)

        self._timeline = timeline
        return timeline

    def _get_label(self, event: TrajectoryEvent) -> str:
        """Get display label for event."""
        icon = self.ICONS.get(event.type, "•")
        type_name = event.type.value.replace("_", " ").title()
        return f"{icon} {type_name}"

    def _format_content(self, event: TrajectoryEvent) -> str:
        """Format event content for display."""
        content = event.content
        if len(content) > 500:
            content = content[:500] + "..."
        return content

    def get_stats(self) -> TrajectoryStats:
        """
        Calculate trajectory statistics.

        Returns:
            TrajectoryStats with analysis
        """
        if not self.events:
            return TrajectoryStats(
                total_events=0,
                total_duration_ms=0,
                max_depth=0,
                events_by_type={},
                tokens_used=0,
                repl_executions=0,
                recursive_calls=0,
            )

        events_by_type: dict[str, int] = {}
        max_depth = 0
        tokens_used = 0
        repl_executions = 0
        recursive_calls = 0

        for event in self.events:
            type_name = event.type.value
            events_by_type[type_name] = events_by_type.get(type_name, 0) + 1
            max_depth = max(max_depth, event.depth)

            # Count specific events
            if event.type == TrajectoryEventType.REPL_EXEC:
                repl_executions += 1
            elif event.type == TrajectoryEventType.RECURSE_START:
                recursive_calls += 1

            # Extract token usage from metadata
            if event.metadata:
                tokens_used += event.metadata.get("tokens", 0)

        # Calculate duration
        start_time = self.events[0].timestamp
        end_time = self.events[-1].timestamp
        total_duration_ms = (end_time - start_time) * 1000

        return TrajectoryStats(
            total_events=len(self.events),
            total_duration_ms=total_duration_ms,
            max_depth=max_depth,
            events_by_type=events_by_type,
            tokens_used=tokens_used,
            repl_executions=repl_executions,
            recursive_calls=recursive_calls,
        )

    def export(self, format: ExportFormat, path: Path | None = None) -> str:
        """
        Export trajectory in specified format.

        Implements: Spec §8.1 Export formats (JSON, HTML)

        Args:
            format: Export format
            path: Optional path to write to

        Returns:
            Exported content as string
        """
        if format == ExportFormat.JSON:
            content = self._export_json()
        elif format == ExportFormat.HTML:
            content = self._export_html()
        else:
            content = self._export_markdown()

        if path:
            path.write_text(content)

        return content

    def _export_json(self) -> str:
        """Export as JSON."""
        data = {
            "stats": {
                "total_events": self.get_stats().total_events,
                "total_duration_ms": self.get_stats().total_duration_ms,
                "max_depth": self.get_stats().max_depth,
            },
            "events": [
                {
                    "timestamp": e.timestamp,
                    "type": e.type.value,
                    "depth": e.depth,
                    "content": e.content,
                    "metadata": e.metadata,
                }
                for e in self.events
            ],
        }
        return json.dumps(data, indent=2)

    def _export_html(self) -> str:
        """Export as interactive HTML."""
        timeline = self.build_timeline()
        stats = self.get_stats()

        events_html = []
        for entry in timeline:
            events_html.append(f"""
        <div class="event depth-{entry.depth}" style="margin-left: {entry.depth * 20}px;">
            <div class="event-header">
                <span class="time">{entry.formatted_time}</span>
                <span class="label">{html.escape(entry.label)}</span>
                {f'<span class="duration">({entry.duration_ms:.1f}ms)</span>' if entry.duration_ms else ''}
            </div>
            <div class="event-content">{html.escape(entry.content)}</div>
        </div>""")

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RLM Trajectory Viewer</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
        }}
        .stats {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}
        .stat {{
            text-align: center;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2563eb;
        }}
        .stat-label {{
            font-size: 12px;
            color: #666;
        }}
        .timeline {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .event {{
            border-left: 3px solid #2563eb;
            padding: 10px 15px;
            margin-bottom: 10px;
            background: #f8fafc;
            border-radius: 0 8px 8px 0;
        }}
        .event:hover {{
            background: #e0f2fe;
        }}
        .event-header {{
            display: flex;
            gap: 10px;
            align-items: center;
            margin-bottom: 5px;
        }}
        .time {{
            font-family: monospace;
            font-size: 12px;
            color: #666;
        }}
        .label {{
            font-weight: 600;
        }}
        .duration {{
            font-size: 12px;
            color: #059669;
        }}
        .event-content {{
            font-family: monospace;
            font-size: 13px;
            white-space: pre-wrap;
            word-break: break-word;
            color: #374151;
        }}
        .depth-0 {{ border-left-color: #2563eb; }}
        .depth-1 {{ border-left-color: #7c3aed; }}
        .depth-2 {{ border-left-color: #db2777; }}
        .controls {{
            margin-bottom: 20px;
        }}
        button {{
            padding: 8px 16px;
            margin-right: 10px;
            border: none;
            border-radius: 4px;
            background: #2563eb;
            color: white;
            cursor: pointer;
        }}
        button:hover {{
            background: #1d4ed8;
        }}
    </style>
</head>
<body>
    <h1>🔍 RLM Trajectory Viewer</h1>

    <div class="stats">
        <div class="stats-grid">
            <div class="stat">
                <div class="stat-value">{stats.total_events}</div>
                <div class="stat-label">Events</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats.total_duration_ms:.0f}ms</div>
                <div class="stat-label">Duration</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats.max_depth}</div>
                <div class="stat-label">Max Depth</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats.repl_executions}</div>
                <div class="stat-label">REPL Executions</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats.recursive_calls}</div>
                <div class="stat-label">Recursive Calls</div>
            </div>
            <div class="stat">
                <div class="stat-value">{stats.tokens_used:,}</div>
                <div class="stat-label">Tokens Used</div>
            </div>
        </div>
    </div>

    <div class="controls">
        <button onclick="filterDepth(0)">Depth 0</button>
        <button onclick="filterDepth(1)">Depth 1</button>
        <button onclick="filterDepth(2)">Depth 2</button>
        <button onclick="filterDepth(-1)">All</button>
    </div>

    <div class="timeline">
        {''.join(events_html)}
    </div>

    <script>
        function filterDepth(depth) {{
            const events = document.querySelectorAll('.event');
            events.forEach(e => {{
                if (depth === -1) {{
                    e.style.display = 'block';
                }} else {{
                    const eventDepth = parseInt(e.className.match(/depth-(\\d+)/)[1]);
                    e.style.display = eventDepth === depth ? 'block' : 'none';
                }}
            }});
        }}
    </script>
</body>
</html>"""

    def _export_markdown(self) -> str:
        """Export as Markdown."""
        timeline = self.build_timeline()
        stats = self.get_stats()

        lines = [
            "# RLM Trajectory",
            "",
            "## Statistics",
            "",
            f"- **Events**: {stats.total_events}",
            f"- **Duration**: {stats.total_duration_ms:.0f}ms",
            f"- **Max Depth**: {stats.max_depth}",
            f"- **REPL Executions**: {stats.repl_executions}",
            f"- **Recursive Calls**: {stats.recursive_calls}",
            f"- **Tokens Used**: {stats.tokens_used:,}",
            "",
            "## Timeline",
            "",
        ]

        for entry in timeline:
            indent = "  " * entry.depth
            duration = f" ({entry.duration_ms:.1f}ms)" if entry.duration_ms else ""
            lines.append(f"{indent}- **{entry.formatted_time}** {entry.label}{duration}")
            if entry.content:
                content_preview = entry.content[:200]
                if len(entry.content) > 200:
                    content_preview += "..."
                lines.append(f"{indent}  ```")
                lines.append(f"{indent}  {content_preview}")
                lines.append(f"{indent}  ```")

        return "\n".join(lines)


class TrajectoryReplayer:
    """
    Replay trajectories step by step.

    Implements: Spec §8.1 Replay capability
    """

    def __init__(self, visualizer: TrajectoryVisualizer):
        """
        Initialize replayer.

        Args:
            visualizer: Visualizer with loaded trajectory
        """
        self.visualizer = visualizer
        self._position = 0
        self._playing = False

    @property
    def total_steps(self) -> int:
        """Total number of steps."""
        return len(self.visualizer.events)

    @property
    def current_position(self) -> int:
        """Current replay position."""
        return self._position

    @property
    def current_event(self) -> TrajectoryEvent | None:
        """Get current event."""
        if 0 <= self._position < len(self.visualizer.events):
            return self.visualizer.events[self._position]
        return None

    @property
    def progress(self) -> float:
        """Progress as fraction 0-1."""
        if self.total_steps == 0:
            return 0.0
        return self._position / self.total_steps

    def step_forward(self) -> TrajectoryEvent | None:
        """
        Move forward one step.

        Returns:
            Next event or None if at end
        """
        if self._position < len(self.visualizer.events) - 1:
            self._position += 1
            return self.current_event
        return None

    def step_backward(self) -> TrajectoryEvent | None:
        """
        Move backward one step.

        Returns:
            Previous event or None if at start
        """
        if self._position > 0:
            self._position -= 1
            return self.current_event
        return None

    def seek(self, position: int) -> TrajectoryEvent | None:
        """
        Seek to specific position.

        Args:
            position: Position to seek to

        Returns:
            Event at position or None if invalid
        """
        if 0 <= position < len(self.visualizer.events):
            self._position = position
            return self.current_event
        return None

    def seek_to_event_type(
        self, event_type: TrajectoryEventType, forward: bool = True
    ) -> TrajectoryEvent | None:
        """
        Seek to next/previous event of type.

        Args:
            event_type: Type to seek to
            forward: Search forward or backward

        Returns:
            Event found or None
        """
        if forward:
            for i in range(self._position + 1, len(self.visualizer.events)):
                if self.visualizer.events[i].type == event_type:
                    self._position = i
                    return self.current_event
        else:
            for i in range(self._position - 1, -1, -1):
                if self.visualizer.events[i].type == event_type:
                    self._position = i
                    return self.current_event
        return None

    def reset(self) -> None:
        """Reset to beginning."""
        self._position = 0

    def get_events_until_now(self) -> list[TrajectoryEvent]:
        """Get all events up to current position."""
        return self.visualizer.events[: self._position + 1]


def visualize_trajectory(
    events: list[TrajectoryEvent] | None = None,
    path: Path | None = None,
    output_format: ExportFormat = ExportFormat.HTML,
    output_path: Path | None = None,
) -> str:
    """
    Convenience function to visualize a trajectory.

    Args:
        events: Events to visualize (or load from path)
        path: Path to load trajectory from
        output_format: Export format
        output_path: Path to write output

    Returns:
        Visualization as string
    """
    viz = TrajectoryVisualizer(events)

    if path:
        viz.load_from_file(path)

    return viz.export(output_format, output_path)


__all__ = [
    "ExportFormat",
    "TimelineEntry",
    "TrajectoryReplayer",
    "TrajectoryStats",
    "TrajectoryVisualizer",
    "visualize_trajectory",
]
