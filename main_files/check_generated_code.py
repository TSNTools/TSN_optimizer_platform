"""Static check of a generated Python file against the REAL library source, with no LLM.

usage: python check_generated_code.py CODE_FILE [--lib LIB_DIR] [--all]

Runs mypy over CODE_FILE with the library directory on MYPYPATH, so every library class
is read from its actual source: a dataclass's real fields, a constructor's real
parameters, a method that exists or does not. It reports only the error kinds that mean
the code will fail or do something other than intended (a missing attribute, a wrong
argument, a value of the wrong type), not strictness noise (missing annotations).
--all prints every mypy error instead.

Nothing here knows the algorithm or the request: it works on any generated file and any
library directory. It needs mypy installed in the interpreter that runs it.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.path.join(os.path.dirname(HERE), "original_tsn_lib_files")

# mypy error codes that point at a real defect. The rest (annotation-unchecked,
# var-annotated, no-untyped-def, ...) only say the code is not fully annotated.
REAL_DEFECTS = {
    "attr-defined",    # reading an attribute or method the class does not have
    "union-attr",      # the same on a value that may be None
    "call-arg",        # wrong number or names of arguments, an argument given twice
    "arg-type",        # an argument of the wrong type
    "call-overload",   # no overload of the callee accepts these arguments
    "name-defined",    # a name that is never defined
    "index",           # indexing something that cannot be indexed that way
    "operator",        # an operator the operand types do not support
    "return-value",    # returning a value of the wrong type
    "misc",            # includes "got multiple values for argument"
    "has-type",        # a name whose type cannot be determined at use
    "valid-type",      # an annotation that is not a type
    "list-item", "dict-item",
}


def definite(msg: str, code: str) -> bool:
    """True when the error is certain to fail at run time or to pass a missing value: an
    attribute or name that does not exist, arguments that do not bind, or None / a maybe-None
    value passed or returned where the declaration does not allow None. Other type
    mismatches (a str where a Literal is declared, container variance) may be harmless."""
    if code in ("attr-defined", "union-attr", "call-arg", "name-defined", "has-type", "valid-type"):
        return True
    if code == "misc" and "multiple values" in msg:
        return True
    got = re.search(r'type "(.*?)"; expected "(.*?)"', msg) or re.search(r'\(got "(.*?)", expected "(.*?)"\)', msg)
    if got and ("None" in got.group(1)) and ("None" not in got.group(2)) and ("Optional" not in got.group(2)):
        return True
    return False


def check(code_file: str, lib_dir: str, show_all: bool = False) -> list[str]:
    """The defect lines mypy reports for code_file, as 'DEFINITE|OTHER line N: message [code] | source'."""
    source = open(code_file, encoding="utf-8").read().splitlines()
    with tempfile.TemporaryDirectory() as work:
        target = os.path.join(work, "generated_impl.py")
        with open(target, "w", encoding="utf-8") as fh:
            fh.write("\n".join(source) + "\n")
        env = dict(os.environ, MYPYPATH=lib_dir)
        done = subprocess.run(
            [sys.executable, "-m", "mypy", "--no-error-summary", "--no-color-output",
             "--show-error-codes", "--follow-imports=silent", "--check-untyped-defs",
             "--ignore-missing-imports", "--no-incremental", "--cache-dir", os.path.join(work, ".c"),
             target],
            capture_output=True, text=True, env=env, timeout=600)
    if "No module named mypy" in done.stderr:
        sys.exit("mypy is not installed for " + sys.executable)
    out = []
    for line in done.stdout.splitlines():
        m = re.match(r".*generated_impl\.py:(\d+): error: (.*?)(?:\s+\[([a-z-]+)\])?$", line)
        if not m:
            continue
        lineno, msg, code = int(m.group(1)), m.group(2), m.group(3) or ""
        if not show_all and code not in REAL_DEFECTS:
            continue
        text = source[lineno - 1].strip() if 0 < lineno <= len(source) else ""
        kind = "DEFINITE" if definite(msg, code) else "OTHER   "
        out.append(f"{kind} line {lineno}: {msg} [{code}] | {text[:110]}")
    return sorted(out, key=lambda x: (not x.startswith("DEFINITE"),))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("code_file")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    found = check(a.code_file, a.lib, a.all)
    n_def = sum(f.startswith("DEFINITE") for f in found)
    print(f"{a.code_file}: {n_def} definite error(s), {len(found) - n_def} other type mismatch(es)")
    for f in found:
        print("  " + f)
