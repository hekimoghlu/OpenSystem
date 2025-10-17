/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
/* <DESC>
 * Basic URL API use.
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

#if !CURL_AT_LEAST_VERSION(7, 62, 0)
#error "this example requires curl 7.62.0 or later"
#endif

int main(void)
{
  CURLU *h;
  CURLUcode uc;
  char *host;
  char *path;

  h = curl_url(); /* get a handle to work with */
  if(!h)
    return 1;

  /* parse a full URL */
  uc = curl_url_set(h, CURLUPART_URL, "http://example.com/path/index.html", 0);
  if(uc)
    goto fail;

  /* extract hostname from the parsed URL */
  uc = curl_url_get(h, CURLUPART_HOST, &host, 0);
  if(!uc) {
    printf("Host name: %s\n", host);
    curl_free(host);
  }

  /* extract the path from the parsed URL */
  uc = curl_url_get(h, CURLUPART_PATH, &path, 0);
  if(!uc) {
    printf("Path: %s\n", path);
    curl_free(path);
  }

  /* redirect with a relative URL */
  uc = curl_url_set(h, CURLUPART_URL, "../another/second.html", 0);
  if(uc)
    goto fail;

  /* extract the new, updated path */
  uc = curl_url_get(h, CURLUPART_PATH, &path, 0);
  if(!uc) {
    printf("Path: %s\n", path);
    curl_free(path);
  }

fail:
  curl_url_cleanup(h); /* free URL handle */
  return 0;
}
