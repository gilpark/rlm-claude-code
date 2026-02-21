#!/usr/bin/env python3
"""CLI entry point for /causal skill.

Usage:
    python scripts/causal_cli.py "analyze src/auth.py --scope security"
    python scripts/causal_cli.py "status --last"
    python scripts/causal_cli.py "tree"
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for src.* imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from skills.causal_router import handle_causal_command


async def main():
    if len(sys.argv) < 2:
        # Default to help
        result = await handle_causal_command("help")
        print(result)
        sys.exit(0)

    # Join all args into a single string
    args_str = " ".join(sys.argv[1:])

    try:
        result = await handle_causal_command(args_str)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
