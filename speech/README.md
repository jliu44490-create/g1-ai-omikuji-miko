
## Prerequisite

Create venv:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
(or by uv:)
```bash
uv venv
uv pip install -r requirements.txt
```

Download tsukuyomi-chan:

```bash
python -m piper \
  --download-model tsukuyomi \
  --download-dir models/tsukuyomi
```