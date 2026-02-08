import os
import sys
import yaml
import subprocess
import time
from pathlib import Path

# --- Configurare ---
# Căile sunt relative la rădăcina proiectului
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_SOURCE = BASE_DIR / "configs" / "exp2_regularized.yaml"
# Folderul nou cerut de tine pentru separare
NEW_SAVE_DIR = "results/verification_run_fp32"
TEMP_CONFIG_PATH = BASE_DIR / "configs" / "temp_verification_config.yaml"


def run_verification():
    print(f"{'=' * 60}")
    print(f"🚀 PORNIRI RERUN DE VERIFICARE: Augmented FP32")
    print(f"📁 Configurație sursă: {CONFIG_SOURCE}")
    print(f"📂 Output folder: {NEW_SAVE_DIR}")
    print(f"{'=' * 60}\n")

    # 1. Citim configurația originală
    if not CONFIG_SOURCE.exists():
        print(f"❌ Eroare: Nu găsesc {CONFIG_SOURCE}")
        return

    with open(CONFIG_SOURCE, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Modificăm directorul de salvare pentru a izola acest run
    print(f"📝 Modificare save_dir -> {NEW_SAVE_DIR}")
    # Structura yaml-ului tău are paths -> save_dir
    if 'paths' not in config:
        config['paths'] = {}
    config['paths']['save_dir'] = NEW_SAVE_DIR

    # Asigurăm aceleași setări critice (doar verificare)
    # config['data']['augmentation'] este deja 'extended' în fișierul original
    # config['training']['precision'] este 'fp32' (implicit sau specificat)

    # 3. Salvăm configurația temporară
    with open(TEMP_CONFIG_PATH, 'w') as f:
        yaml.dump(config, f)
    print(f"💾 Configurație temporară salvată în: {TEMP_CONFIG_PATH}")

    # 4. Rulăm antrenarea
    print("\n⏳ Începe antrenarea... (Acest proces va dura câteva ore)")
    print("   Te rog să nu închizi terminalul sau laptopul.")
    print(f"{'-' * 60}")

    start_time = time.time()

    # Apelăm scriptul principal de antrenare cu noul config
    # Presupunem că rulezi din root, deci python scripts/train.py
    cmd = [sys.executable, "scripts/train.py", "--config", str(TEMP_CONFIG_PATH)]

    try:
        # Folosim cwd=BASE_DIR pentru a fi siguri că rulăm din root
        result = subprocess.run(cmd, cwd=BASE_DIR, check=True)

        duration = (time.time() - start_time) / 3600
        print(f"\n✅ VERIFICARE FINALIZATĂ în {duration:.2f} ore.")
        print(f"📊 Rezultatele sunt în: {BASE_DIR / NEW_SAVE_DIR}")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ EROARE CRITICĂ în timpul antrenării: {e}")
    finally:
        # Curățenie: ștergem config-ul temporar (opțional, eu l-aș lăsa pt debug)
        # if TEMP_CONFIG_PATH.exists():
        #     os.remove(TEMP_CONFIG_PATH)
        pass


if __name__ == "__main__":
    run_verification()