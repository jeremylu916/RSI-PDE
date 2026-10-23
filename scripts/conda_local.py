"""Run the installed conda with its environment catalog confined to the project.

Conda 25.5.1 touches ~/.conda/environments.txt during transaction verification even
when register_envs=false. Redirect that catalog before importing transaction code.
This adapts installed conda in this process only; no system files are modified.
"""

from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if project_root != Path("/depot/lin491/data/Binghang_project/PDE-RSI"):
    raise SystemExit("The conda wrapper must stay in the PDE-RSI depot project")

from conda.core import envs_manager

catalog = project_root / ".runtime/conda/environments.txt"
if not catalog.resolve().is_relative_to(project_root):
    raise SystemExit("Conda environment catalog escapes project")
envs_manager.get_user_environments_txt_file = lambda userhome="~": str(catalog)

# The wrapper supports only project environment operations and read-only inspection.
arguments = sys.argv[1:]
if not arguments or arguments[0] not in {"create", "install", "list", "info", "env"}:
    raise SystemExit("Use this wrapper for project create/install/list/info/env operations only")
mutation = arguments[0] in {"create", "install"} or (
    arguments[0] == "env" and len(arguments) > 1 and arguments[1] in {"create", "update"})
if arguments[0] == "env" and (len(arguments) < 2 or arguments[1] not in {"create", "update", "export"}):
    raise SystemExit("Supported env operations: create, update, export")
if mutation:
    if "--prefix" not in arguments:
        raise SystemExit("Environment operations require an explicit project --prefix")
    index = arguments.index("--prefix")
    if index+1 >= len(arguments):
        raise SystemExit("Missing --prefix value")
    prefix = Path(arguments[index+1]).resolve()
    if not prefix.is_relative_to(project_root / "envs"):
        raise SystemExit("Conda prefix must remain in the project's envs directory")

from conda.cli import main

raise SystemExit(main(*arguments))
