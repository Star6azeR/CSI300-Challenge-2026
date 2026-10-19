"""Verify or package the standalone paper materials (Python standard library)."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    root = Path(root).resolve()
    index = json.loads((root / 'evidence/index.json').read_text())
    checked = 0
    for items in index['evidence'].values():
        for item in items:
            relative = Path(item['path'])
            target = (root / relative).resolve()
            assert not relative.is_absolute() and target.is_relative_to(root), item['path']
            assert target.is_file() and digest(target) == item['sha256'], item['path']
            checked += 1
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    patterns = {
        'machine_path': r'/Users/[A-Za-z0-9._-]+/|/home/[A-Za-z0-9._-]+/|/private/(?:tmp|var)/[A-Za-z0-9._-]+|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+',
        'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'credential': r'(?i)(?:api[_-]?key|password|access[_-]?token|client[_-]?secret|authorization)\s*[=:]\s*[\"\']?(?:sk-[A-Za-z0-9_-]{15,}|Bearer\s+[A-Za-z0-9_.-]{16,}|[A-Za-z0-9_+/=-]{24,})',
        'access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,})\b',
        'email': r'\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
    }
    findings = []
    for name, record in manifest.items():
        target = (root / name).resolve()
        assert target.is_relative_to(root) and target.is_file(), name
        assert digest(target) == record['sha256'] and target.stat().st_size == record['bytes'], name
        assert not any(x in target.parts for x in ['.git', '.venv', '__pycache__']), name
        assert target.suffix.lower() not in {'.parquet', '.pkl', '.npz', '.pt', '.pth', '.cbm', '.env'}, name
        if target.suffix.lower() == '.pdf':
            continue
        text = target.read_text(encoding='utf-8')
        for category, pattern in patterns.items():
            if re.search(pattern, text):
                findings.append({'path': name, 'category': category})
    assert not findings, findings
    return {'evidence_references': checked, 'files_checked': len(manifest),
            'relative_paths_resolve': True, 'hashes_match': True, 'scan_findings': findings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify the write-up source and evidence files.')
    parser.add_argument('action', choices=['verify'])
    parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
