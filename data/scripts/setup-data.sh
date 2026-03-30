#!/usr/bin/env bash
# Download and set up the SkillFlow corpus.
# Run from the data/ directory:  bash scripts/setup-data.sh
set -euo pipefail

URL="https://ucdavis.box.com/shared/static/7zr4wsaaydclt89rgt78n0di1fnned55.zip"
ZIP="data.zip"

echo "Downloading corpus..."
curl -L -o "$ZIP" "$URL"

echo "Extracting..."
unzip -qo "$ZIP"

echo "Reconstructing skills-no-letta/skillsmp..."
cp -r skills/skillsmp skills-no-letta/skillsmp

rm "$ZIP"
echo "Done. Corpus ready at skills/ ($(ls skills/skillsmp | wc -l) skills)."
