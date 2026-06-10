import urllib.request
import json
import sys

# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8')

try:
    url = 'https://zenodo.org/api/records/10688648'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as r:
        data = json.loads(r.read())
        for f in data.get('files', []):
            print(f.get('key', f.get('filename')), f['size'] / (1024*1024), "MB")
except Exception as e:
    print("Error:", e)
