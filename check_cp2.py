import csv
import re
from pathlib import Path

# Thư mục kiểm tra theo Checkpoint 2
D = Path('data/academic_regulations')

REQ = ['doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience']
mds = sorted(D.glob('*.md'))
sources_file = D / 'sources.csv'

if not sources_file.exists():
    print(f"Lỗi: Không tìm thấy {sources_file}")
    exit(1)

rows = list(csv.DictReader(open(sources_file, encoding='utf-8')))
ids, auds = [], {}

for p in mds:
    content = p.read_text(encoding='utf-8')
    parts = content.split('---')
    if len(parts) >= 3:
        fm = dict(re.findall(r'^(\w+):\s*(.+)$', parts[1], re.M))
    else:
        fm = {}
    
    doc_id = fm.get('doc_id')
    ids.append(doc_id)
    audience = fm.get('audience')
    auds[audience] = auds.get(audience, 0) + 1
    
    status = "OK" if all(k in fm for k in REQ) and doc_id == p.stem else "THIEU METADATA"
    print(f'{p.name:40} {status}')

print('so file :', len(mds), '(can 5-10)')
print('csv     :', 'khop' if sorted(r['doc_id'] for r in rows) == sorted(ids) else 'LECH')
print('audience:', auds)
