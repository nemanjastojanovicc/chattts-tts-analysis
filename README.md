# ChatTTS TTS

Notebook generiše govor pomoću ChatTTS-a i analizira trajanje, RTF, signalne karakteristike i subjektivne ocene.

## Pokretanje

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name chattts-tts --display-name "Python (chattts-tts)"
jupyter notebook analiza.ipynb
```
