# -*- coding: utf-8 -*-
"""核验 v2 manifest 钉住的 v1 文件 hash。

冻结说明第三节的硬约束：v2 引用必须验 hash。规则：
- `stability="frozen"`：v1 已物理只读，hash 不一致即 ERROR。
- `stability="snapshot"`：v1 该目录仍可能被其它会话写入，hash 不一致只 WARN（说明变动，不静默放行）。

用法：python tests/verify_upstream.py [--json]
退出码：0 全部通过（允许 WARN）；1 存在 ERROR。
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(os.path.dirname(HERE), "manifest.json")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv):
    as_json = "--json" in argv
    with open(MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)

    base = manifest["upstream"]["base"]
    checks = []

    archive_rel = manifest["upstream"]["archive_manifest"]
    archive_path = os.path.join(base, archive_rel.replace("/", os.sep))
    entry = {"path": archive_rel, "stability": "frozen",
             "expect": manifest["upstream"]["archive_manifest_sha256"]}
    entry["actual"] = sha256_file(archive_path) if os.path.isfile(archive_path) else None
    checks.append(entry)

    for item in manifest["upstream"]["files"]:
        path = os.path.join(base, item["path"].replace("/", os.sep))
        actual = sha256_file(path) if os.path.isfile(path) else None
        checks.append({"path": item["path"], "stability": item["stability"],
                       "expect": item["sha256"], "actual": actual})

    errors, warns, ok = [], [], 0
    for c in checks:
        if c["actual"] == c["expect"]:
            ok += 1
        elif c["stability"] == "frozen":
            errors.append(c)
        else:
            warns.append(c)

    if as_json:
        print(json.dumps({"ok": ok, "errors": errors, "warns": warns}, ensure_ascii=False, indent=1))
    else:
        for c in checks:
            mark = "OK  " if c["actual"] == c["expect"] else ("WARN" if c["stability"] == "snapshot" else "FAIL")
            print("%s %s" % (mark, c["path"]))
        print("---")
        print("通过 %d / 总计 %d；ERROR %d，WARN %d" % (ok, len(checks), len(errors), len(warns)))
        for c in errors:
            print("  ERROR %s 期望 %s 实际 %s" % (c["path"], c["expect"][:16], (c["actual"] or "缺失")[:16]))
        for c in warns:
            print("  WARN  %s 冻结后已变动（该目录仍在写）" % c["path"])

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
