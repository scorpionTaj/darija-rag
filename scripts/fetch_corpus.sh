#!/usr/bin/env bash
# Download the corpus from the Ministry of Justice portal (adala.justice.gov.ma) into data/raw/.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"

base="https://adala.justice.gov.ma/api/uploads"
fetch() {  # fetch <output name> <path under $base>
  local encoded
  encoded=$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$2")
  [[ -s "$1" ]] && { echo "have $1"; return; }
  curl -fsSL --retry 3 -o "$1.part" "$base/$encoded" && mv "$1.part" "$1"
  echo "got  $1"
}

fetch labour_code_ar.pdf "2026/07/28/ظهير شريف رقم 1.03.194 بتنفيذ القانون رقم 65.99 المتعلق بمدونة الشغل -1785240385771.pdf"
fetch labour_code_fr.pdf "2024/04/30/code du travail-1714463246806.pdf"
fetch family_code_ar.pdf "2024/07/22/مدونة الأسرة-1721646687560.pdf"
fetch family_code_fr.pdf "2024/12/12/DAHIR N° 1-04-22 PORTANT PROMULGATION DE LA LOI.PDF-1734012188728.pdf"
fetch rental_law_ar.pdf "2026/04/15/ظهير شريف رقم 1.13.111 صادر في-1776242288111.pdf"
fetch obligations_contracts_ar.pdf "2024/10/24/ظهير بمثابة قانون الالتزامات والعقود-1729777400387.pdf"
fetch obligations_contracts_fr.pdf "2024/02/28/Code des Obligations et des Contrats_compressed-1709126934943.pdf"

shasum -a 256 -c ../SHA256SUMS
