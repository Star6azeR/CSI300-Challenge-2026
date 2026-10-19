from pathlib import Path
import json
import hashlib

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def sanitize(value):
    if isinstance(value, dict):
        return {sanitize(str(k)): sanitize(v) for k, v in value.items() if k not in ('authorized_attempt_id', 'ple_training_provenance', 'external_training_window')}
    if isinstance(value, list):
        return [sanitize(x) for x in value]
    if isinstance(value, str) and (value.startswith('/') or ':\\' in value):
        return Path(value.replace('\\', '/')).name
    return value

def manifest(model_dir):
    write(model_dir / 'manifest.json', {'format_version': 1, 'files_sha256': {str(path.relative_to(model_dir)): sha(path) for path in sorted(model_dir.rglob('*')) if path.is_file() and path.name != 'manifest.json'}})
