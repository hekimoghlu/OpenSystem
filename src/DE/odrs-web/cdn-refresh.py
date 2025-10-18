#/usr/bin/python

import os
import sys
import time

import requests

token = os.environ["CDN77_TOKEN"]
if not token:
    sys.exit()

cdn_id = os.environ["CDN77_ID"]
if not cdn_id:
    sys.exit()

base_url = f"https://api.cdn77.com/v3/cdn/{cdn_id}"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}",
}

data = '{"paths":["/1.0/reviews/api/ratings"]}'
purge = requests.post(f"{base_url}/job/purge", headers=headers, data=data)

if purge_id := purge.json().get("id"):
    purge_status = requests.get(f"{base_url}/job/{purge_id}", headers=headers)
    while purge_status.json().get("state") != "done":
        time.sleep(1)
        purge_status = requests.get(f"{base_url}/job/{purge_id}", headers=headers)

prefetch = requests.post(f"{base_url}/job/prefetch", headers=headers, data=data)
