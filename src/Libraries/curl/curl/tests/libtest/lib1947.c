/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 14, 2024.
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

static size_t writecb(char *data, size_t n, size_t l, void *userp)
{
  /* ignore the data */
  (void)data;
  (void)userp;
  return n*l;
}
int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  struct curl_header *h;
  int count = 0;
  unsigned int origins;

  global_init(CURL_GLOBAL_DEFAULT);

  easy_init(curl);

  /* perform a request that involves redirection */
  easy_setopt(curl, CURLOPT_URL, URL);
  easy_setopt(curl, CURLOPT_WRITEFUNCTION, writecb);
  easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "curl_easy_perform() failed: %s\n",
            curl_easy_strerror(res));
    goto test_cleanup;
  }

  /* count the number of requests by reading the first header of each
     request. */
  origins = (CURLH_HEADER|CURLH_TRAILER|CURLH_CONNECT|
             CURLH_1XX|CURLH_PSEUDO);
  do {
    h = curl_easy_nextheader(curl, origins, count, NULL);
    if(h)
      count++;
  } while(h);
  printf("count = %u\n", count);

  /* perform another request - without redirect */
  easy_setopt(curl, CURLOPT_URL, libtest_arg2);
  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "curl_easy_perform() failed: %s\n",
            curl_easy_strerror(res));
    goto test_cleanup;
  }

  /* count the number of requests again. */
  count = 0;
  do {
    h = curl_easy_nextheader(curl, origins, count, NULL);
    if(h)
      count++;
  } while(h);
  printf("count = %u\n", count);

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();
  return (int)res;
}
