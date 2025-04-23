import subprocess
import os

def run_script(script_name):
    script_path = os.path.join(script_name)
    print(f"\n===== Running {script_name} =====")
    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(f"[ERROR] {script_name} stderr:\n{result.stderr}")

if __name__ == "__main__":
    scripts = [
        "save_image.py",
        "yolopose.py",
        "feature_extract.py",
        "local_clustering.py",
        "global_clustering.py"
    ]

    for script in scripts:
        run_script(script)

    print("\n>>> 전체 파이프라인 실행 완료.")
