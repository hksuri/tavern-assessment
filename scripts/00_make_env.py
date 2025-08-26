
#!/usr/bin/env python3
import sys, platform, pkgutil
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Installed (subset):")
for p in ["pandas","numpy","scipy","sklearn","statsmodels","matplotlib","joblib"]:
    try:
        m = __import__(p if p!="sklearn" else "sklearn")
        print(f" - {p}: OK")
    except Exception as e:
        print(f" - {p}: MISSING ({e})")
