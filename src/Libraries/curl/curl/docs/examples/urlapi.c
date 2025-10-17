/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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
 * Set working URL with CURLU *.
 * </DESC>
 */
#include <stdio.h>
#include <curl/curl.h>

#if !CURL_AT_LEAST_VERSION(7, 80, 0)
#error "this example requires curl 7.80.0 or later"
#endif

int main(void)
{
  CURL *curl;
  CURLcode res;

  CURLU *urlp;
  CURLUcode uc;

  /* get a curl handle */
  curl = curl_easy_init();

  /* init Curl URL */
  urlp = curl_url();
  uc = curl_url_set(urlp, CURLUPART_URL,
                    "http://example.com/path/index.html", 0);

  if(uc) {
    fprintf(stderr, "curl_url_set() failed: %s", curl_url_strerror(uc));
    goto cleanup;
  }

  if(curl) {
    /* set urlp to use as working URL */
    curl_easy_setopt(curl, CURLOPT_CURLU, urlp);
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    /* only allow HTTP, TFTP and SFTP */
    curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, "http,tftp,sftp");

    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
      fprintf(stderr, "curl_easy_perform() failed: %s\n",
              curl_easy_strerror(res));

    goto cleanup;
  }

cleanup:
  curl_url_cleanup(urlp);
  curl_easy_cleanup(curl);
  return 0;
}
