# %% [Step 1] Mount Google Drive & Environment Setup
"""
Phase 09: Google Colab Model Bundle Export & Verification Notebook
"""
import os, sys, hashlib, json, zipfile, subprocess
from pathlib import Path

# Step 1: Install Dependencies
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic('pip', 'install -q torch safetensors lightgbm scipy pandas numpy statsmodels pydantic scikit-learn pillow')
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch", "safetensors", "lightgbm", "scipy", "pandas", "numpy", "statsmodels", "pydantic", "scikit-learn", "pillow"], check=False)
except Exception:
    pass

# Step 2: Workspace & Import Paths Setup
def setup_colab_workspace():
    if "google.colab" in sys.modules:
        if not os.path.exists("/content/drive"):
            try:
                from google.colab import drive
                drive.mount("/content/drive")
            except Exception:
                pass
    
    candidates = [
        os.getcwd(),
        "/content/PlantDiseaseSeverity",
        "/content/crop-prognosis",
        "/content/drive/MyDrive/PlantDiseaseSeverity",
        "/content/drive/MyDrive/crop-prognosis"
    ]
    
    # Check all subdirectories in /content for ml/export
    content_dir = Path("/content")
    if content_dir.exists():
        try:
            for p in content_dir.glob("**/ml/export"):
                repo_candidate = str(p.parent.parent.resolve())
                if repo_candidate not in candidates:
                    candidates.insert(0, repo_candidate)
        except Exception:
            pass

    found_repo = False
    for path in candidates:
        if os.path.exists(path) and os.path.exists(os.path.join(path, "ml")):
            if path not in sys.path:
                sys.path.insert(0, path)
            os.chdir(path)
            print(f"[INFO] Using project repository root at: {path}")
            found_repo = True
            break
            
    if not found_repo:
        print("[WARNING] Could not automatically locate 'ml' package root. Searching parent directories...")
        current = Path.cwd().resolve()
        for parent in [current] + list(current.parents):
            if (parent / "ml" / "export").exists():
                p_str = str(parent)
                if p_str not in sys.path:
                    sys.path.insert(0, p_str)
                os.chdir(parent)
                print(f"[INFO] Dynamically resolved repo root to: {parent}")
                found_repo = True
                break

    # If still not found in Colab, automatically clone project repository
    if not found_repo and "google.colab" in sys.modules:
        print("[INFO] Repository not found locally. Cloning repository into /content/PlantDiseaseSeverity...")
        target_clone = "/content/PlantDiseaseSeverity"
        subprocess.run(["git", "clone", "https://github.com/nesan-s/PlantDiseaseSeverity.git", target_clone], check=False)
        if os.path.exists(os.path.join(target_clone, "ml")):
            if target_clone not in sys.path:
                sys.path.insert(0, target_clone)
            os.chdir(target_clone)
            print(f"[INFO] Successfully cloned repository and set working directory to: {target_clone}")
            found_repo = True

setup_colab_workspace()

# Import Phase 09 export module
from ml.export.export_bundle import export_bundle
from ml.export.verify_bundle import verify_bundle

# %% [Step 2] Run Export Bundle Builder
print("===========================================================================")
print("STARTING MODEL BUNDLE EXPORT PIPELINE...")
print("===========================================================================")

raw_dir = Path("data/raw")
output_zip = raw_dir / "crop-prognosis-bundle-v1.0.0.zip"

bundle_zip_path = export_bundle(
    data_raw_dir=raw_dir,
    output_zip_path=output_zip,
    bundle_version="1.0.0"
)

# %% [Step 3] Calculate SHA-256 Checksum & Run Verifier
def calculate_file_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

zip_sha256 = calculate_file_sha256(bundle_zip_path)
print("===========================================================================")
print("EXPORT COMPLETE:")
print(f"  - Zip File Location : {bundle_zip_path}")
print(f"  - File Size         : {bundle_zip_path.stat().st_size / (1024*1024):.2f} MB")
print(f"  - SHA-256 Checksum  : {zip_sha256}")
print("===========================================================================")

# Run verifier directly on exported zip
passed, report_lines = verify_bundle(bundle_zip_path)
print("\n".join(report_lines))

# %% [Step 4] Download Zip File (Google Colab only)
if "google.colab" in sys.modules:
    from google.colab import files
    print("[INFO] Triggering automatic browser download of model bundle zip...")
    files.download(str(bundle_zip_path))
else:
    print(f"[INFO] Local export ready. Download or copy file from: {bundle_zip_path}")
