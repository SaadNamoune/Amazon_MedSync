"""
Windows compatibility shim, imported before any nvflare import.

nvflare/fuel/f3/cellnet/net_agent.py does an unconditional `import resource`
at module load time. `resource` is POSIX-only (used there just for an admin
`process_info` diagnostic command this project never calls) -- without this
shim, `import nvflare` crashes immediately on native Windows Python, which
is otherwise a fully supported way to run everything else in this repo.
"""
import sys
import types

if sys.platform == "win32" and "resource" not in sys.modules:
    _stub = types.ModuleType("resource")
    _stub.RUSAGE_SELF = 0
    _stub.getrusage = lambda who: types.SimpleNamespace(ru_maxrss=0)
    sys.modules["resource"] = _stub
