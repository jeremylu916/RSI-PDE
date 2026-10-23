#!/usr/bin/env bash
# Rebuild the RSI-PDE Bench website from the current aggregates and publish it to the gh-pages branch of
# github.com/jeremylu916/RSI-PDE (GitHub Pages: https://jeremylu916.github.io/RSI-PDE/).
# Usage (login node, after sourcing scripts/env.sh): bash scripts/publish_site.sh
set -euo pipefail
root=/depot/lin491/data/Binghang_project/PDE-RSI
cd "$root"
python scripts/bench.py report > /dev/null
python scripts/build_site.py
pages=$root/.runtime/publish/pages
if [[ ! -d $pages/.git ]]; then
  git clone -q --branch gh-pages https://github.com/jeremylu916/RSI-PDE.git "$pages"
fi
git -C "$pages" pull -q --ff-only origin gh-pages
cp site/index.html "$pages/index.html"; touch "$pages/.nojekyll"
if git -C "$pages" diff --quiet; then echo "site unchanged"; exit 0; fi
git -C "$pages" add -A
git -C "$pages" commit -q -m "Update the RSI-PDE Bench website

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git -C "$pages" push -q origin gh-pages
echo "published https://jeremylu916.github.io/RSI-PDE/"
