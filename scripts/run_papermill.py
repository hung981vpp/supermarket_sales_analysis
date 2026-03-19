import sys
import os
import papermill as pm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocess_feature.ipynb",
    "notebooks/03_mining_or_clustering.ipynb",
    "notebooks/04_modeling.ipynb",
    "notebooks/05_evaluation_report.ipynb",
]

OUTPUT_DIR = Path("notebooks/runs")

def run_all_notebooks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    Path("outputs/tables").mkdir(parents=True, exist_ok=True)

    for nb_path in NOTEBOOKS:
        nb_name = Path(nb_path).name
        out_path = OUTPUT_DIR / nb_name
        print(f"\n▶ Running {nb_name}...")
        pm.execute_notebook(
            input_path=nb_path,
            output_path=str(out_path),
            kernel_name="python3",
            cwd=str(ROOT),          # ← ép CWD về root
        )
        print(f"✅ Done → {out_path}")

    print("\n🎉 All notebooks executed successfully.")

if __name__ == "__main__":
    run_all_notebooks()
