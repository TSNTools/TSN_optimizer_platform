"""Mechanical checks of a design blueprint (the pipeline's blueprint stage output), with no LLM.

usage: python check_blueprint.py BLUEPRINT_FILE [--lib LIB_DIR]

BLUEPRINT_FILE holds the blueprint JSON (text around the JSON object is ignored): its
"data_structures", "function_inventory" (name, signature, calls, reads, writes) and
"reuse_decisions". Every check is about the design's own consistency, never about what the
algorithm should do, so it works for any request:

  CYCLE      functions that call each other in a loop with no way out in the design
  MISSING    a function calls another that needs a library object (a class of the library,
             directly or through a structure that is just that class), and the caller has no
             way to get it: it is not one of its parameters (or held inside one), no function it
             calls returns it, and it does not build it (its "writes", or a library constructor
             its reuse decision gives it)
  UNDEFINED  a signature names a type that is not a data structure, not a library class
             and not a Python/typing name
  UNKNOWN    a "calls" entry that is neither a function of the inventory nor a reused
             library member
  UNCALLED   a function no other function reaches from the top-level function
  NONAME     a "reads"/"writes" entry that is not a data structure name

Library class names are read from the library source (every `class` in LIB_DIR).
"""
import argparse
import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LIB = os.path.join(os.path.dirname(HERE), "original_tsn_lib_files")

PY_NAMES = {
    "int", "float", "bool", "str", "bytes", "None", "object", "type", "list", "dict", "tuple",
    "set", "frozenset", "complex", "Optional", "Union", "Literal", "Any", "Callable", "Iterable",
    "Iterator", "Sequence", "Mapping", "MutableMapping", "TypeAlias", "np", "numpy", "ndarray",
    "NDArray", "typing", "List", "Dict", "Tuple", "Set", "FrozenSet",
}
SCALARS = {"int", "float", "bool", "str", "bytes", "None", "complex"}


def library_classes(lib_dir):
    names = set()
    for fn in os.listdir(lib_dir):
        if fn.endswith(".py"):
            try:
                tree = ast.parse(open(os.path.join(lib_dir, fn), encoding="utf-8").read())
            except SyntaxError:
                continue
            names |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    return names


def load_blueprint(path):
    text = open(path, encoding="utf-8").read()
    return json.loads(text[text.find("{"): text.rfind("}") + 1])


def split_params(sig):
    inner = sig[sig.find("(") + 1: sig.rfind(")")] if "(" in sig else ""
    out, depth, cur = [], 0, ""
    for ch in inner:
        depth += ch in "[("
        depth -= ch in "])"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    params = []
    for p in out:
        p = p.strip()
        if not p or p in ("self", "*", "/") or p.startswith("**") or p.startswith("*"):
            continue
        name, _, typ = p.partition(":")
        params.append((name.strip(), typ.split("=")[0].strip()))
    return params


def return_type(sig):
    return sig.rsplit("->", 1)[-1].strip() if "->" in sig else ""


def names_in(type_text):
    type_text = re.sub(r"'[^']*'|\"[^\"]*\"", "", type_text or "")   # Literal values are not types
    return {n.split(".")[-1] for n in re.findall(r"[A-Za-z_][\w.]*", type_text)}


def check(bp, lib_classes):
    structs = {s.get("name"): (s.get("type") or "") for s in bp.get("data_structures") or [] if isinstance(s, dict)}
    funcs = {f["name"]: f for f in bp.get("function_inventory") or [] if isinstance(f, dict) and f.get("name")}
    reused = bp.get("reuse_decisions") or []
    reused_names = {str(r.get("codebase_function", "")).split(".")[-1] for r in reused if isinstance(r, dict)}
    issues = []

    def expand(type_names):
        """every type reachable inside these types through the data structures' definitions"""
        seen, todo = set(), list(type_names)
        while todo:
            t = todo.pop()
            if t in seen:
                continue
            seen.add(t)
            if t in structs:
                todo.extend(names_in(structs[t]))
        return seen

    def canonical(t, depth=0):
        """a structure whose type is just another name is that name (StartSchedule: Schedule -> Schedule)"""
        if depth > 10 or t not in structs:
            return t
        text = structs[t].split(":")[0].split(" - ")[0].strip()   # "Schedule: what it holds" -> "Schedule"
        if re.fullmatch(r"[A-Za-z_][\w.]*", text):
            return canonical(text.split(".")[-1], depth + 1)
        return t

    def scalar_struct(t):
        """a structure that is only a plain value (an int, a str, a Literal...): any function can make one"""
        if t not in structs:
            return False
        inner = names_in(structs[t]) - {"Literal", "Optional", "Union"}
        return bool(inner) and inner <= SCALARS

    def builds(caller):
        """types the caller makes itself: its writes, and library classes whose constructor it reuses"""
        made = set(funcs[caller].get("writes") or [])
        for r in reused:
            if not isinstance(r, dict) or str(r.get("used_by", "")) != caller:
                continue
            cls = str(r.get("class_name") or "")
            if cls and str(r.get("codebase_function", "")).split(".")[-1] in ("__init__", "__new__"):
                made.add(cls)
        return made

    # callers of each function, and the top-level function
    called = {c for f in funcs.values() for c in (f.get("calls") or [])}
    tops = [n for n in funcs if n not in called]
    top = tops[0] if tops else next(iter(funcs), None)

    # CYCLE
    cycles = set()
    def dfs(n, path):
        for c in funcs.get(n, {}).get("calls") or []:
            if c in path:
                loop = path[path.index(c):]
                k = min(range(len(loop)), key=lambda i: loop[i])
                cycles.add(" -> ".join(loop[k:] + loop[:k] + [loop[k]]))
            elif c in funcs:
                dfs(c, path + [c])
    for n in funcs:
        dfs(n, [n])
    issues += [("CYCLE", c) for c in sorted(cycles)]

    # MISSING
    for n, f in funcs.items():
        have = {t for _, ty in split_params(f.get("signature", "")) for t in names_in(ty)}
        for c in f.get("calls") or []:
            if c in funcs:
                have |= names_in(return_type(funcs[c].get("signature", "")))
        have |= builds(n)
        have = {canonical(t) for t in expand(have)}
        for c in f.get("calls") or []:
            if c not in funcs:
                continue
            for pname, ty in split_params(funcs[c].get("signature", "")):
                # only library objects: a function cannot make a problem, a use case, an evaluator or a
                # schedule out of nothing; the design's own records and maps it can build from what it holds
                needed = {canonical(t) for t in names_in(ty)} & lib_classes
                for t in sorted(needed - have):
                    issues.append(("MISSING", f"{n} calls {c}, which needs {pname}: {t}; {n} has no {t} "
                                              f"(not a parameter, not returned by its calls, not built by it)"))

    # UNDEFINED
    for n, f in funcs.items():
        sig = f.get("signature", "")
        used = set()
        for _, ty in split_params(sig):
            used |= names_in(ty)
        used |= names_in(return_type(sig))
        for t in sorted(used - set(structs) - lib_classes - PY_NAMES):
            issues.append(("UNDEFINED", f"{n}: type {t} is not a data structure, a library class or a Python type"))

    # UNKNOWN
    for n, f in funcs.items():
        for c in f.get("calls") or []:
            if c not in funcs and c.split(".")[-1] not in reused_names:
                issues.append(("UNKNOWN", f"{n} calls {c}, which is neither in the inventory nor a reused library member"))

    # UNCALLED
    if top:
        reach, todo = set(), [top]
        while todo:
            x = todo.pop()
            if x in reach:
                continue
            reach.add(x)
            todo += [c for c in funcs.get(x, {}).get("calls") or [] if c in funcs]
        for n in funcs:
            if n not in reach:
                issues.append(("UNCALLED", f"{n} is never reached from the top-level function {top}"))

    # NONAME
    for n, f in funcs.items():
        for key in ("reads", "writes"):
            for s in f.get(key) or []:
                if isinstance(s, str) and s not in structs:
                    issues.append(("NONAME", f"{n} {key} {s!r}, which is not a data structure name"))

    out, seen = [], set()
    for kind, text in issues:
        if (kind, text) not in seen:
            seen.add((kind, text))
            out.append((kind, text))
    return top, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("blueprint_file")
    ap.add_argument("--lib", default=DEFAULT_LIB)
    a = ap.parse_args()
    top, found = check(load_blueprint(a.blueprint_file), library_classes(a.lib))
    print(f"{a.blueprint_file}: top-level {top}, {len(found)} problem(s)")
    order = ["CYCLE", "MISSING", "UNDEFINED", "UNKNOWN", "UNCALLED", "NONAME"]
    for kind, text in sorted(found, key=lambda x: order.index(x[0])):
        print(f"  {kind:9} {text}")
