"""Mirror a public zarr surface-volume locally with per-object retries.

Exists because the async streaming reader in the ink inference path aborts the whole
run on a single truncated chunk read (ClientPayloadError / SSL RECORD_LAYER_FAILURE),
while the same objects fetch 100% reliably over plain sequential HTTPS.
"""
import sys, os, time, urllib.request, urllib.error
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BUCKET = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_keys(prefix):
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        with urllib.request.urlopen(url, timeout=120) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{NS}Contents"):
            keys.append(c.find(f"{NS}Key").text)
        t = root.find(f"{NS}NextContinuationToken")
        if t is None or not t.text:
            break
        token = t.text
    return keys


def fetch(key, outroot, prefix, tries=6):
    rel = key[len(prefix):]
    dst = os.path.join(outroot, rel.replace("/", os.sep))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    last = None
    for a in range(tries):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{key}", timeout=180) as r:
                data = r.read()
                declared = r.headers.get("Content-Length")
            if declared is not None and len(data) != int(declared):
                raise IOError(f"short read {len(data)}/{declared}")
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return len(data)
        except Exception as e:
            last = e
            time.sleep(min(2 ** a, 20))
    print(f"  PERMANENT FAIL {rel}: {last}", flush=True)
    return -1


def main(prefix, outroot):
    keys = list_keys(prefix)
    print(f"{len(keys)} objects under {prefix}", flush=True)
    os.makedirs(outroot, exist_ok=True)
    done = bad = 0
    total = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n in ex.map(lambda k: fetch(k, outroot, prefix), keys):
            if n < 0:
                bad += 1
            else:
                total += n
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(keys)}  {total/1e6:.1f} MB", flush=True)
    print(f"done {done} objects, {total/1e6:.1f} MB, {bad} permanent failures", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
