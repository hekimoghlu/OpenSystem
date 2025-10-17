/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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
 * This unit test PUT http data over proxy. Proxy header will be different
 * from server http header
 */

#include "test.h"

#include "memdebug.h"

static char data [] = "Hello Cloud!\n";

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t  amount = nmemb * size; /* Total bytes curl wants */
  if(amount < strlen(data)) {
    return strlen(data);
  }
  (void)stream;
  memcpy(ptr, data, strlen(data));
  return strlen(data);
}

int test(char *URL)
{
  CURL *curl = NULL;
  CURLcode res = CURLE_FAILED_INIT;
  /* http and proxy header list */
  struct curl_slist *hhl = NULL, *phl = NULL, *tmp = NULL;

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

  hhl = curl_slist_append(hhl, "User-Agent: Http Agent");
  phl = curl_slist_append(phl, "User-Agent: Proxy Agent");
  if(!hhl || !phl) {
    goto test_cleanup;
  }
  tmp = curl_slist_append(phl, "Expect:");
  if(!tmp) {
    goto test_cleanup;
  }
  phl = tmp;

  test_setopt(curl, CURLOPT_URL, URL);
  test_setopt(curl, CURLOPT_PROXY, libtest_arg2);
  test_setopt(curl, CURLOPT_HTTPHEADER, hhl);
  test_setopt(curl, CURLOPT_PROXYHEADER, phl);
  test_setopt(curl, CURLOPT_HEADEROPT, CURLHEADER_SEPARATE);
  test_setopt(curl, CURLOPT_POST, 0L);
  test_setopt(curl, CURLOPT_UPLOAD, 1L);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_PROXYTYPE, CURLPROXY_HTTP);
  test_setopt(curl, CURLOPT_HEADER, 1L);
  test_setopt(curl, CURLOPT_WRITEFUNCTION, fwrite);
  test_setopt(curl, CURLOPT_READFUNCTION, read_callback);
  test_setopt(curl, CURLOPT_HTTPPROXYTUNNEL, 1L);
  test_setopt(curl, CURLOPT_INFILESIZE, (long)strlen(data));

  res = curl_easy_perform(curl);

test_cleanup:

  curl_easy_cleanup(curl);

  curl_slist_free_all(hhl);

  curl_slist_free_all(phl);

  curl_global_cleanup();

  return (int)res;
}
