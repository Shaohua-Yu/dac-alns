"""Check all files in the small release using the standard library."""
from pathlib import Path
import hashlib
import json


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'checksums.json').read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        p = (root / row['path']).resolve()
        if not p.is_relative_to(root):
            errors.append('Invalid manifest path')
            continue
        if not p.is_file():
            errors.append('Missing: ' + row['path'])
            continue
        if p.stat().st_size != row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            errors.append('Changed: ' + row['path'])
    print(json.dumps({'valid': not errors, 'checked_files': len(manifest['files']), 'errors': errors}, ensure_ascii=True))
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
