# Keystroke Auth

РЎРёСЃС‚РµРјР° СЂР°СЃРїРѕР·РЅР°РІР°РЅРёСЏ РєР»Р°РІРёР°С‚СѓСЂРЅРѕРіРѕ РїРѕС‡РµСЂРєР° РїРѕР»СЊР·РѕРІР°С‚РµР»РµР№ РєР°Рє СЃСЂРµРґСЃС‚РІРѕ Р±РёРѕРјРµС‚СЂРёС‡РµСЃРєРѕР№ Р°СѓС‚РµРЅС‚РёС„РёРєР°С†РёРё.

## Р‘С‹СЃС‚СЂС‹Р№ СЃС‚Р°СЂС‚

`powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
`
"@ | Set-Content -Encoding UTF8 "keystroke_auth/README.md"

@"
# Roadmap

1. РџРѕРґРіРѕС‚РѕРІРєР° СЃС‚СЂСѓРєС‚СѓСЂС‹ РїСЂРѕРµРєС‚Р°.
2. Р—Р°РіСЂСѓР·РєР° CMU Keystroke Dynamics Benchmark.
3. РР·РІР»РµС‡РµРЅРёРµ РїСЂРёР·РЅР°РєРѕРІ: hold, press-press, release-release, release-press, press-release.
4. РќРѕСЂРјР°Р»РёР·Р°С†РёСЏ Рё split.
5. РћР±СѓС‡РµРЅРёРµ MLP baseline.
6. Accuracy, confusion matrix, FAR, FRR, EER.
7. Embedding-Р°СѓС‚РµРЅС‚РёС„РёРєР°С†РёСЏ Рё enrollment.
