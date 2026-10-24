#!/bin/sh
# Download / verify the ANP "Serie Historica de Precos de Combustiveis" semi-annual automotive
# fuel files (dsas/ca). URL list: external/data/anp_brazil_fuel/anp_ca_urls.txt (scraped from
# https://www.gov.br/anp/pt-br/centrais-de-conteudo/dados-abertos/serie-historica-de-precos-de-combustiveis
# on 2026-10-23). gov.br answers HEAD with 403, so the remote size is read from a 1-byte range
# request (Content-Range). Partial files are resumed with curl -C -. The sandbox proxy drops
# connections when too many are open, so parallelism is kept low and retries are generous.
set -u
D=/workspace/projects/TenDeFlAn/external/data/anp_brazil_fuel
exec 9>"$D/.fetch.lock"; flock -n 9 || { echo "fetch_anp.sh is already running (lock $D/.fetch.lock); not starting another"; exit 0; }
cd "$D/raw" || exit 1
PAR=${PAR:-3}
fetch_one() {
  u="$1"; f=$(basename "$u")
  remote=$(curl -sS -m 60 -L -r 0-0 -D - -o /dev/null --retry 5 --retry-all-errors --retry-delay 3 "$u" 2>/dev/null \
           | tr -d '\r' | awk 'tolower($1)=="content-range:"{split($0,a,"/"); n=a[2]} END{print n+0}')
  local=$( [ -f "$f" ] && stat -c %s "$f" || echo 0 )
  if [ "$remote" -gt 0 ] && [ "$remote" = "$local" ]; then echo "OK   $f $local"; return; fi
  # only trust a remote size that looks like a real file (an error page or a failed probe must not delete data)
  if [ "$remote" -gt 1000000 ] && [ "$local" -gt "$remote" ]; then mv -f -- "$f" "$f.oversize.bak"; fi
  curl -sS -m 7200 -L -C - --retry 8 --retry-all-errors --retry-delay 5 -o "$f" "$u" 2>/dev/null
  local=$( [ -f "$f" ] && stat -c %s "$f" || echo 0 )
  if [ "$remote" -gt 0 ] && [ "$remote" = "$local" ]; then echo "GOT  $f $local"; else echo "FAIL $f local=$local remote=$remote"; fi
}
# URLS may point to a reordered list (e.g. control-window years first); default: all files in date order
for u in $(cat "${URLS:-$D/anp_ca_urls.txt}"); do fetch_one "$u" &
  while [ "$(jobs -r | wc -l)" -ge "$PAR" ]; do sleep 1; done
done; wait
ls | wc -l; du -sh .
