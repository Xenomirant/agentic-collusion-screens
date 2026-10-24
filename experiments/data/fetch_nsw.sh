#!/bin/sh
# Download NSW FuelCheck monthly price-history files (data.nsw.gov.au, dataset "fuel-check").
# URL list: external/data/nsw_fuelcheck/nsw_urls.txt (CKAN API, 2026-10-23). Skips files already
# present with the server's Content-Length.
set -u
D=/workspace/projects/TenDeFlAn/external/data/nsw_fuelcheck
cd "$D/raw" || exit 1
fetch_one() {
  u="$1"; f="$(echo "$u" | sed -E 's#.*/resource/([0-9a-f]{8})[^/]*/download/(.*)#\1_\2#')"
  remote=$(curl -sSI -m 60 -L "$u" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}')
  local=$( [ -f "$f" ] && stat -c %s "$f" || echo 0 )
  if [ "$remote" -gt 0 ] && [ "$remote" = "$local" ]; then echo "OK   $f $local"; return; fi
  curl -sS -m 1800 -L --retry 8 --retry-all-errors --retry-delay 5 -o "$f" "$u" && echo "GOT  $f $(stat -c %s "$f") (remote $remote)" || echo "FAIL $f"
}
for u in $(cat "$D/nsw_urls.txt"); do fetch_one "$u" &
  while [ "$(jobs -r | wc -l)" -ge 2 ]; do sleep 1; done
done; wait
ls | wc -l; du -sh .
