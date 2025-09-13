#!/bin/bash
# flatten.sh
# Moves all files from subdirectories into one folder called "flat" inside the current directory.
# Duplicate files are renamed with a numeric suffix.

set -e

DEST="./flat"

# Create destination directory if it doesn't exist
mkdir -p "$DEST"

# Counter for duplicates
declare -A seen

# Find and move files
find . -mindepth 2 -type f | while read -r file; do
    base=$(basename "$file")
    target="$DEST/$base"

    # If file with same name already exists, append counter
    if [[ -e "$target" ]]; then
        count=1
        while [[ -e "$DEST/${base%.*}_$count.${base##*.}" ]]; do
            ((count++))
        done
        target="$DEST/${base%.*}_$count.${base##*.}"
    fi

    mv "$file" "$target"
done

echo "✅ All files moved into: $DEST"

