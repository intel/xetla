#!/bin/bash
set -e
echo [Format Checking] GITHUB_BASE_REF $GITHUB_BASE_REF
echo [Format Checking] GITHUB_HEAD_REF $GITHUB_HEAD_REF
echo [Format Checking] GITHUB_SHA $GITHUB_SHA
echo [Format Checking] Changed files
git diff --name-only origin/${GITHUB_BASE_REF} origin/${GITHUB_HEAD_REF}
echo [Format Checking] Checking...
git diff --name-only origin/${GITHUB_BASE_REF} origin/${GITHUB_HEAD_REF} | (grep -v "3rd_parties" | grep -E "\.(cpp|c|h|hpp)$" || true) | xargs -i sh -c "clang-format-10 {} | cmp {}"
echo [Format Checking] Finished