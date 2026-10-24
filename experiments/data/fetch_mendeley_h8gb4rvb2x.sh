#!/bin/sh
# Parallel range download of the Mendeley Data file "2022-FY-combined.dta"
# (dataset h8gb4rvb2x v1 "2022 Daily German Retail Fuel Price Data", 2,532,830,924 bytes,
# derived from Tankerkoenig history, CC BY-NC-SA 4.0). The S3 backend serves ~200 KB/s per
# connection but supports byte ranges, so we fetch 48 chunks, PAR (default 3) at a time, and concatenate. The VPN proxy
# also carries the assistant's own API traffic: keep at most 3 downloads through it at once (all fetchers combined).
set -u
OUT=/workspace/projects/TenDeFlAn/external/data/tankerkoenig/raw
mkdir -p "$OUT"
exec 9>"$OUT/../.fetch.lock"; flock -n 9 || { echo "fetch_mendeley is already running (lock $OUT/../.fetch.lock); not starting another"; exit 0; }
LANDING='https://data.mendeley.com/public-files/datasets/h8gb4rvb2x/files/3230a714-5798-4a5a-8bfc-3e153d28f4d8/file_downloaded'
S3=$(curl -sSI -m 60 "$LANDING" | tr -d '\r' | awk 'tolower($1)=="location:"{print $2}' | tail -1)
[ -z "$S3" ] && S3='https://prod-dcd-datasets-public-files-eu-west-1.s3.eu-west-1.amazonaws.com/cf28bdea-de87-4eb4-afe7-03431f49fea7'
echo "S3=$S3"
SIZE=2532830924
N=48
PAR=${PAR:-3}
CH=$(( (SIZE + N - 1) / N ))
mkdir -p "$OUT/parts"
i=0
while [ $i -lt $N ]; do
  s=$((i*CH)); e=$(( (i+1)*CH - 1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
  p="$OUT/parts/part_$(printf %03d $i)"; want=$((e-s+1))
  if [ -f "$p" ] && [ "$(stat -c %s "$p")" = "$want" ]; then :; else
    ( curl -sS -m 3000 --retry 8 --retry-all-errors --retry-delay 5 -r $s-$e -o "$p" "$S3" || echo "FAIL $i" ) &
    while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do sleep 1; done
  fi
  i=$((i+1))
done
wait
ok=0; i=0; while [ $i -lt $N ]; do s=$((i*CH)); e=$(( (i+1)*CH - 1 )); [ $e -ge $SIZE ] && e=$((SIZE-1)); p="$OUT/parts/part_$(printf %03d $i)"; [ -f "$p" ] && [ "$(stat -c %s "$p")" = "$((e-s+1))" ] && ok=$((ok+1)); i=$((i+1)); done
echo "complete parts: $ok / $N"
if [ $ok -eq $N ]; then
  cat $(ls "$OUT"/parts/part_* | sort) > "$OUT/2022-FY-combined.dta"
  ls -la "$OUT/2022-FY-combined.dta"
  [ "$(stat -c %s "$OUT/2022-FY-combined.dta")" = "$SIZE" ] && echo SIZE_OK || echo SIZE_MISMATCH
fi
