"""Layer-2 smoke test: import every src.* module in an isolated subprocess.
Scripts with side-effects-at-import-time will run their main() — that's
itself a bug worth reporting, so we use --help via -m where possible
to avoid that. For pure libraries (no top-level main), the import alone
is the check."""
import os, subprocess, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

PY = r"C:\Users\z0502\anaconda3\envs\navlm_v2\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["NAVLM_DATA"] = r"C:\Users\z0502\Desktop\cs231n\navlm_v2\data"

# Modules with argparse: invoke `python -m src.X --help` (lightweight, no side effects)
# Pure libraries: just import them.
HAS_ARGPARSE = set()
for fn in os.listdir(os.path.join(HERE, "src")):
    if not fn.endswith(".py") or fn == "__init__.py": continue
    src = open(os.path.join(HERE, "src", fn), encoding="utf-8").read()
    if "argparse" in src or 'if __name__' in src:
        HAS_ARGPARSE.add(fn[:-3])

modules = sorted(f[:-3] for f in os.listdir(os.path.join(HERE, "src"))
                  if f.endswith(".py") and f != "__init__.py")

results = []
for m in modules:
    use_help = m in HAS_ARGPARSE
    if use_help:
        cmd = [PY, "-m", f"src.{m}", "--help"]
    else:
        cmd = [PY, "-c", f"import src.{m}; print('OK')"]
    try:
        r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=30)
        ok = (r.returncode == 0) or (use_help and "usage" in (r.stdout + r.stderr).lower())
        err = ""
        if not ok:
            err = (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else f"exit {r.returncode}"
        results.append((m, use_help, ok, err[:140]))
    except subprocess.TimeoutExpired:
        results.append((m, use_help, False, "TIMEOUT (30s)"))
    except Exception as e:
        results.append((m, use_help, False, f"{type(e).__name__}: {e}"))

n_ok = sum(1 for _, _, ok, _ in results if ok)
print(f"\n=== smoke-test results: {n_ok}/{len(results)} OK ===\n")
print(f"  {'module':<32s} {'mode':<8s} {'status':<6s}  msg")
print(f"  {'-'*100}")
for m, use_help, ok, err in results:
    mode = "--help" if use_help else "import"
    status = "OK" if ok else "FAIL"
    print(f"  {m:<32s} {mode:<8s} {status:<6s}  {err}")
