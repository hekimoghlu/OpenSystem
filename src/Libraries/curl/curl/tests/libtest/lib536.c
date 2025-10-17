/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 23, 2024.
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
#include "test.h"

#include "memdebug.h"

#define WITH_PROXY     "http://usingproxy.com/"
#define WITHOUT_PROXY  libtest_arg2

static void proxystat(CURL *curl)
{
  long wasproxy;
  if(!curl_easy_getinfo(curl, CURLINFO_USED_PROXY, &wasproxy)) {
    printf("This %sthe proxy\n", wasproxy ? "used ":
           "DID NOT use ");
  }
}

int test(char *URL)
{
  CURLcode res = CURLE_OK;
  CURL *curl;
  struct curl_slist *host = NULL;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  host = curl_slist_append(NULL, libtest_arg3);
  if(!host)
    goto test_cleanup;

  test_setopt(curl, CURLOPT_RESOLVE, host);
  test_setopt(curl, CURLOPT_PROXY, URL);
  test_setopt(curl, CURLOPT_URL, WITH_PROXY);
  test_setopt(curl, CURLOPT_NOPROXY, "goingdirect.com");
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  res = curl_easy_perform(curl);
  if(!res) {
    proxystat(curl);
    test_setopt(curl, CURLOPT_URL, WITHOUT_PROXY);
    res = curl_easy_perform(curl);
    if(!res)
      proxystat(curl);
  }

test_cleanup:

  curl_easy_cleanup(curl);
  curl_slist_free_all(host);
  curl_global_cleanup();

  return (int)res;
}
