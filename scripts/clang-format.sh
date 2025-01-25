#!/bin/bash

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Function to format files with clang-format
format_files() {
    local dir=$1
    local ext=$2
    
    # Find all files with the specified extension in the directory
    find "$dir" -type f -name "*.$ext" | while read -r file; do
        echo "Formatting: $file"
        clang-format -i -style=file "$file"
    done
}

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format is not installed"
    echo "Please install clang-format first:"
    echo "  Ubuntu/Debian: sudo apt-get install clang-format"
    echo "  macOS: brew install clang-format"
    exit 1
fi

# Format all C++ source files
echo "Formatting C++ source files..."
format_files "$PROJECT_ROOT/benchmark" "cu"
format_files "$PROJECT_ROOT/benchmark" "h"

echo "Code formatting complete!"

# Optionally check for formatting differences
if [ "$1" == "--check" ]; then
    echo "Checking for formatting differences..."
    git diff --exit-code
    if [ $? -eq 0 ]; then
        echo "No formatting issues found."
    else
        echo "Formatting issues found. Please commit the changes."
        exit 1
    fi
fi
