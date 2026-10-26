#!/usr/bin/env python
"""Static check: names REFERENCED but never BOUND in any enclosing scope.
This is exactly the bug class that killed a 27-job campaign (a string-surgery
edit silently no-opped, leaving `_old_norm_mean` referenced but unassigned --
AST-parseable, crashes at runtime). Uses stdlib symtable: no dependencies.

Run: python tests/test_static.py   (exits nonzero on findings)
"""
import builtins, os, sys, symtable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["train.py", "evaluate.py", "diagnostic.py", "analysis.py"] + \
    [os.path.join("scsrc", f) for f in os.listdir(os.path.join(ROOT, "scsrc")) if f.endswith(".py")] + \
    [os.path.join("scripts", f) for f in os.listdir(os.path.join(ROOT, "scripts")) if f.endswith(".py")]

BUILTIN = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__class__"}  # __class__: implicit cell of super()


def walk(table, bound_above, findings, path):
    bound = set(bound_above)
    for sym in table.get_symbols():
        if sym.is_assigned() or sym.is_imported() or sym.is_parameter() or sym.is_declared_global():
            bound.add(sym.get_name())
    for sym in table.get_symbols():
        n = sym.get_name()
        # NOTE: unbound names in functions are classified GLOBAL_IMPLICIT, so we
        # must NOT exclude is_global(); module-level bindings arrive via bound_above.
        if sym.is_referenced() and not sym.is_assigned() and not sym.is_imported() \
                and not sym.is_parameter() and n not in bound_above and n not in BUILTIN \
                and not sym.is_declared_global():
            findings.append(f"{path}: scope '{table.get_name()}': '{n}' referenced but never bound")
    for child in table.get_children():
        walk(child, bound, findings, path)


def main():
    findings = []
    for rel in FILES:
        fp = os.path.join(ROOT, rel)
        if not os.path.exists(fp):
            continue
        src = open(fp).read()
        try:
            st = symtable.symtable(src, rel, "exec")
        except SyntaxError as e:
            findings.append(f"{rel}: SyntaxError {e}")
            continue
        walk(st, set(), findings, rel)
        # 2026-09-23 (v1.125): a module-level function defined twice silently takes the LATER body. An edit that
        # replaced one block left an older heldout_question() below it, which overrode the new one unnoticed.
        import re as _re
        _defs = _re.findall(r"^def ([A-Za-z_][A-Za-z_0-9]*)\(", src, _re.M)
        for _d in sorted({d for d in _defs if _defs.count(d) > 1}):
            findings.append(f"{rel}: module-level function '{_d}' defined {_defs.count(_d)} times (the last one wins)")
        # 2026-09-24 (v1.139): a function-local import of a name that is ALSO imported at module level makes the name
        # local to the whole function, so every use outside that branch raises UnboundLocalError. train.py's ladder
        # preflight did exactly this ("from scsrc.dataset import ObjectiveDataset" inside main) and killed all 48
        # ladder main-grid training tasks at start-up (2026-09-23 16:20-23:17).
        import ast as _ast
        try:
            _tree = _ast.parse(src)
        except SyntaxError:
            _tree = None
        if _tree is not None:
            _top = set()
            for _n in _tree.body:
                if isinstance(_n, (_ast.Import, _ast.ImportFrom)):
                    _top |= {(_a.asname or _a.name).split(".")[0] for _a in _n.names}
            for _fn in _ast.walk(_tree):
                if isinstance(_fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                    _direct = {id(x) for x in _fn.body}           # an unconditional import at the top level of the body is safe
                    for _n in _ast.walk(_fn):
                        if isinstance(_n, (_ast.Import, _ast.ImportFrom)) and id(_n) not in _direct:
                            for _a in _n.names:
                                _nm = (_a.asname or _a.name).split(".")[0]
                                if _nm in _top:
                                    findings.append(f"{rel}:{_n.lineno}: '{_nm}' imported inside {_fn.name}() shadows the "
                                                    f"module-level import -> UnboundLocalError on other paths")
    if findings:
        print("STATIC CHECK FAILURES:")
        for f in findings:
            print("  ", f)
        sys.exit(1)
    print(f"STATIC CHECK PASSED ({len(FILES)} files: no unresolved names)")


if __name__ == "__main__":
    main()
