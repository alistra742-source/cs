#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────
# Kaggle session setup — run as ONE cell, from the freshly cloned repo.
# Idempotent: safe to re-run after a kill/restart.
#
#   !cd /kaggle/working/cs && bash kaggle_setup.sh
#
# What it does (~2 min + photo fetch/copy):
#   1. real hCaptcha vehicle tiles   (orlov clone, once)
#   2. real hCaptcha tiles in repo   (hcap_real/ merge, always)
#   3. generated roster set          (hcap_gen/: 4 photoreal -> photos/,
#                                      2 hCaptcha-style flats -> hcap/)
#   4. real photos for the rest      (brain-photos dataset if mounted,
#                                      else fetch_photos.py from Wikimedia)
# ─────────────────────────────────────────────────────────────────────────
set -e
cd /kaggle/working/cs

echo "── 1/4 real hCaptcha vehicle tiles ──"
if [ ! -d /kaggle/working/hcap ]; then
    git clone --depth 1 https://github.com/orlov-ai/hcaptcha-dataset \
        /kaggle/working/hcap
fi

echo "── 2/4 repo-shipped real hCaptcha tiles ──"
cp -rn hcap_real/* /kaggle/working/hcap/ 2>/dev/null || true

echo "── 3/4 generated roster set (13 classes, ~70/30 photoreal/flat) ──"
for d in hcap_gen/*/; do
    c=$(basename "$d")
    case "$c" in _*) continue ;; esac
    mkdir -p "/kaggle/working/hcap/$c" "/kaggle/working/photos/$c"
    # flats (recognisable hCaptcha-style) -> tile-crop path
    cp -n "$d"photo_5.jpg "$d"photo_6.jpg "/kaggle/working/hcap/$c/" \
        2>/dev/null || true
    # photoreal -> centre-crop path
    cp -n "$d"photo_1.jpg "$d"photo_2.jpg "$d"photo_3.jpg "$d"photo_4.jpg \
        "/kaggle/working/photos/$c/" 2>/dev/null || true
done

echo "── 4/4 real photos for the other ~970 classes ──"
if [ -d /kaggle/input/brain-photos/photos ]; then
    echo "brain-photos dataset mounted - copying (fast, no fetch)"
    cp -rn /kaggle/input/brain-photos/photos/* /kaggle/working/photos/ \
        2>/dev/null || true
else
    echo "no brain-photos dataset - fetching from Wikimedia (~15-40 min)"
    python fetch_photos.py --out /kaggle/working/photos --per_class 4
fi

echo "── setup done ──"
echo "hcap tiles: $(find /kaggle/working/hcap -type f | wc -l) files"
echo "photos:     $(find /kaggle/working/photos -type f | wc -l) files"
echo ""
echo "ONE-TIME (after this session's fetch, ~15 min): zip the photos folder"
echo "to /kaggle/output, download it from the Output tab and upload it as a"
echo "Kaggle dataset named brain-photos (folder 'photos' at its root)."
echo "Every later session then skips the fetch entirely:"
echo "    !cd /kaggle/working/photos && zip -qr /kaggle/output/photos.zip ."
