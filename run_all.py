import subprocess
import sys
from pathlib import Path

EPOCHS = "5"

for task in ["fruit", "ripeness", "defect"]:
    if not Path("dataset", task, "train").exists():
        subprocess.run([
            sys.executable, "-m", "src.prepare_dataset", "split",
            "--source", "raw_data/train",
            "--test-source", "raw_data/test",
            "--task", task,
            "--mapping", f"mappings/{task}.json",
        ], check=True)

    subprocess.run([
        sys.executable, "-m", f"src.train_{task}",
        "--epochs", EPOCHS,
        "--fine-tune-epochs", "0",
    ], check=True)

print("All 3 models trained. Now run: streamlit run app.py")
EOF