#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./build_cache.sh 12
#   ./build_cache.sh 14
#   ./build_cache.sh 16
# Env overrides:
#   MONGO_HOSTPORT=192.168.50.171:27017
#   MONGO_DB=ECDSA_BTC
#   MONGO_COLLECTION=BTC
#   MONGO_FIELD=r
#   OUT_FILE=mongo_keys.cache.bin
#   BATCH_SIZE=200000

HEX_LEN="${1:-12}"
MONGO_HOSTPORT="${MONGO_HOSTPORT:-192.168.50.171:27017}"
MONGO_DB="${MONGO_DB:-ECDSA_BTC}"
MONGO_COLLECTION="${MONGO_COLLECTION:-BTC}"
MONGO_FIELD="${MONGO_FIELD:-r}"
OUT_FILE="${OUT_FILE:-mongo_keys.cache.bin}"
BATCH_SIZE="${BATCH_SIZE:-200000}"

if [[ "$HEX_LEN" != "12" && "$HEX_LEN" != "14" && "$HEX_LEN" != "16" ]]; then
  echo "HEX_LEN must be 12, 14, or 16"
  exit 1
fi

if [[ ! -x ./GenCacheMongo ]]; then
  echo "GenCacheMongo not found, building..."
  g++ -O3 -std=c++17 -o GenCacheMongo GenCacheMongo.cpp $(pkg-config --cflags --libs libmongoc-1.0)
fi

echo "Start cache build"
echo "  mongo:      $MONGO_HOSTPORT"
echo "  db:         $MONGO_DB"
echo "  collection: $MONGO_COLLECTION"
echo "  field:      $MONGO_FIELD"
echo "  hex-len:    $HEX_LEN"
echo "  out:        $OUT_FILE"

./GenCacheMongo \
  --mongo "$MONGO_HOSTPORT" \
  --db "$MONGO_DB" \
  --collection "$MONGO_COLLECTION" \
  --field "$MONGO_FIELD" \
  --hex-len "$HEX_LEN" \
  --batch-size "$BATCH_SIZE" \
  --out "$OUT_FILE"
