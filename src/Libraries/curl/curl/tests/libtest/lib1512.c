/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
/*
 * Use global DNS cache (while deprecated it should still work), populate it
 * with CURLOPT_RESOLVE in the first request and then make sure a subsequent
 * easy transfer finds and uses the populated stuff.
 */

#include "test.h"

#include "memdebug.h"

#define NUM_HANDLES 2

int test(char *URL)
{
  int res = 0;
  CURL *curl[NUM_HANDLES] = {NULL, NULL};
  char *port = libtest_arg3;
  char *address = libtest_arg2;
  char dnsentry[256];
  struct curl_slist *slist = NULL;
  int i;
  char target_url[256];
  (void)URL; /* URL is setup in the code */

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  msnprintf(dnsentry, sizeof(dnsentry), "server.example.curl:%s:%s",
            port, address);
  printf("%s\n", dnsentry);
  slist = curl_slist_append(slist, dnsentry);

  /* get NUM_HANDLES easy handles */
  for(i = 0; i < NUM_HANDLES; i++) {
    /* get an easy handle */
    easy_init(curl[i]);
    /* specify target */
    msnprintf(target_url, sizeof(target_url),
              "http://server.example.curl:%s/path/1512%04i",
              port, i + 1);
    target_url[sizeof(target_url) - 1] = '\0';
    easy_setopt(curl[i], CURLOPT_URL, target_url);
    /* go verbose */
    easy_setopt(curl[i], CURLOPT_VERBOSE, 1L);
    /* include headers */
    easy_setopt(curl[i], CURLOPT_HEADER, 1L);

    CURL_IGNORE_DEPRECATION(
      easy_setopt(curl[i], CURLOPT_DNS_USE_GLOBAL_CACHE, 1L);
    )
  }

  /* make the first one populate the GLOBAL cache */
  easy_setopt(curl[0], CURLOPT_RESOLVE, slist);

  /* run NUM_HANDLES transfers */
  for(i = 0; (i < NUM_HANDLES) && !res; i++) {
    res = curl_easy_perform(curl[i]);
    if(res)
      goto test_cleanup;
  }

test_cleanup:

  curl_easy_cleanup(curl[0]);
  curl_easy_cleanup(curl[1]);
  curl_slist_free_all(slist);
  curl_global_cleanup();

  return res;
}
