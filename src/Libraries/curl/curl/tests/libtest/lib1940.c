/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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

static const char *show[]={
  "daTE",
  "Server",
  "content-type",
  "content-length",
  "location",
  "set-cookie",
  "silly-thing",
  "fold",
  "blank",
  "Blank2",
  NULL
};

#ifdef LIB1946
#define HEADER_REQUEST 0
#else
#define HEADER_REQUEST -1
#endif

static void showem(CURL *easy, unsigned int type)
{
  int i;
  struct curl_header *header;
  for(i = 0; show[i]; i++) {
    if(CURLHE_OK == curl_easy_header(easy, show[i], 0, type, HEADER_REQUEST,
                                     &header)) {
      if(header->amount > 1) {
        /* more than one, iterate over them */
        size_t index = 0;
        size_t amount = header->amount;
        do {
          printf("- %s == %s (%u/%u)\n", header->name, header->value,
                 (int)index, (int)amount);

          if(++index == amount)
            break;
          if(CURLHE_OK != curl_easy_header(easy, show[i], index, type,
                                           HEADER_REQUEST, &header))
            break;
        } while(1);
      }
      else {
        /* only one of this */
        printf(" %s == %s\n", header->name, header->value);
      }
    }
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
  CURL *easy = NULL;
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_DEFAULT);
  easy_init(easy);
  easy_setopt(easy, CURLOPT_URL, URL);
  easy_setopt(easy, CURLOPT_VERBOSE, 1L);
  easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
  /* ignores any content */
  easy_setopt(easy, CURLOPT_WRITEFUNCTION, write_cb);

  /* if there's a proxy set, use it */
  if(libtest_arg2 && *libtest_arg2) {
    easy_setopt(easy, CURLOPT_PROXY, libtest_arg2);
    easy_setopt(easy, CURLOPT_HTTPPROXYTUNNEL, 1L);
  }
  res = curl_easy_perform(easy);
  if(res)
    goto test_cleanup;

  showem(easy, CURLH_HEADER);
  if(libtest_arg2 && *libtest_arg2) {
    /* now show connect headers only */
    showem(easy, CURLH_CONNECT);
  }
  showem(easy, CURLH_1XX);
  showem(easy, CURLH_TRAILER);

test_cleanup:
  curl_easy_cleanup(easy);
  curl_global_cleanup();
  return (int)res;
}
