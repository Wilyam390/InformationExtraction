#!/usr/bin/env bash
set -euo pipefail

# Downloads Kaggle invoice images, prepares local OCR source, rebuilds dataset,
# and retrains the classifier.

DATASET_SLUG="${DATASET_SLUG:-osamahosamabdellatif/high-quality-invoice-images-for-ocr}"
LOCAL_INVOICE_IMAGE_MAX="${LOCAL_INVOICE_IMAGE_MAX:-1500}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"
DOWNLOAD_DIR="${RAW_DIR}/kaggle_invoice_download"
INVOICE_IMAGE_DIR="${RAW_DIR}/invoice_kaggle_images"

echo "==> Root: ${ROOT_DIR}"
echo "==> Dataset: ${DATASET_SLUG}"
echo "==> Local OCR image cap: ${LOCAL_INVOICE_IMAGE_MAX}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI is not installed."
  echo "Install with: pip install kaggle"
  exit 1
fi

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "ERROR: Kaggle credentials not found at ~/.kaggle/kaggle.json"
  echo "Create an API token in Kaggle Account settings and place it there."
  exit 1
fi

mkdir -p "${RAW_DIR}"
rm -rf "${DOWNLOAD_DIR}" "${INVOICE_IMAGE_DIR}"
mkdir -p "${DOWNLOAD_DIR}" "${INVOICE_IMAGE_DIR}"

echo "==> Downloading Kaggle dataset..."
kaggle datasets download \
  --dataset "${DATASET_SLUG}" \
  --path "${DOWNLOAD_DIR}" \
  --unzip \
  --force

echo "==> Collecting image files into ${INVOICE_IMAGE_DIR}..."
python3 - "${DOWNLOAD_DIR}" "${INVOICE_IMAGE_DIR}" <<'PY'
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
count = 0

for p in src.rglob("*"):
    if p.is_file() and p.suffix.lower() in exts:
        out = dst / f"{count:06d}{p.suffix.lower()}"
        shutil.copy2(p, out)
        count += 1

print(f"Copied {count} images to {dst}")
if count == 0:
    raise SystemExit("No images found in downloaded dataset.")
PY

echo "==> Rebuilding dataset with local invoice OCR source..."
(
  cd "${ROOT_DIR}"
  INVOICE_IMAGE_DIR="${INVOICE_IMAGE_DIR}" \
  LOCAL_INVOICE_IMAGE_MAX="${LOCAL_INVOICE_IMAGE_MAX}" \
  python3 src/build_dataset.py
)

echo "==> Retraining model..."
(
  cd "${ROOT_DIR}"
  python3 src/train.py
)

echo "==> Done."
echo "Models updated under: ${ROOT_DIR}/models"
