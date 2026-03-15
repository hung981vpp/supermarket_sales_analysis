import sys
import papermill as pm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

NOTEBOOKS = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_preprocess_feature.ipynb",
    "notebooks/03_mining_or_clustering.ipynb",
    "notebooks/04_modeling.ipynb",
    "notebooks/05_evaluation_report.ipynb",
]

# Lưu notebook đã chạy xong vào notebooks/runs/.
OUTPUT_DIR = Path("notebooks/runs")


def run_all_notebooks():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for nb_path in NOTEBOOKS:
        nb_name  = Path(nb_path).name
        out_path = OUTPUT_DIR / nb_name
        print(f"\n▶ Running {nb_name}...")

        pm.execute_notebook(
            input_path=nb_path,
            output_path=str(out_path),
            kernel_name="python3",
        )
        print(f"✅ Done → {out_path}")

    print("\n🎉 All notebooks executed successfully.")
    print(f"   Output notebooks saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_all_notebooks()
