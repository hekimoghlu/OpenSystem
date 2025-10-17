/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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

#ifdef _MSC_VER
/* warning C4706: assignment within conditional expression */
#pragma warning(disable:4706)
#endif
static void showem(CURL *easy, unsigned int type)
{
  struct curl_header *header = NULL;
  struct curl_header *prev = NULL;

  while((header = curl_easy_nextheader(easy, type, 0, prev))) {
    printf(" %s == %s (%u/%u)\n", header->name, header->value,
           (int)header->index, (int)header->amount);
    prev = header;
  }
}

static size_t write_cb(char *data, size_t n, size_t l, void *userp)
{
  /* take care of the data here, ignored in this example */
  (void)data;
  (void)userp;
  return n*l;
}
int test(char *URL)
{
  CURL *easy;
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_DEFAULT);

  easy_init(easy);
  curl_easy_setopt(easy, CURLOPT_URL, URL);
  curl_easy_setopt(easy, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
  /* ignores any content */
  curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_cb);

  /* if there's a proxy set, use it */
  if(libtest_arg2 && *libtest_arg2) {
    curl_easy_setopt(easy, CURLOPT_PROXY, libtest_arg2);
    curl_easy_setopt(easy, CURLOPT_HTTPPROXYTUNNEL, 1L);
  }
  res = curl_easy_perform(easy);
  if(res) {
    printf("badness: %d\n", (int)res);
  }
  showem(easy, CURLH_CONNECT|CURLH_HEADER|CURLH_TRAILER|CURLH_1XX);

test_cleanup:
  curl_easy_cleanup(easy);
  curl_global_cleanup();
  return (int)res;
}
