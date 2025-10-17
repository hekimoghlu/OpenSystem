/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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
 * This source code is used for lib1502, lib1503, lib1504 and lib1505 with
 * only #ifdefs controlling the cleanup sequence.
 *
 * Test case 1502 converted from bug report #3575448, identifying a memory
 * leak in the CURLOPT_RESOLVE handling with the multi interface.
 */

#include "test.h"

#include <limits.h>

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

#define TEST_HANG_TIMEOUT 60 * 1000

int test(char *URL)
{
  CURL *easy = NULL;
  CURL *dup;
  CURLM *multi = NULL;
  int still_running;
  int res = 0;

  char redirect[160];

  /* DNS cache injection */
  struct curl_slist *dns_cache_list;

  res_global_init(CURL_GLOBAL_ALL);
  if(res) {
    return res;
  }

  msnprintf(redirect, sizeof(redirect), "google.com:%s:%s", libtest_arg2,
            libtest_arg3);

  start_test_timing();

  dns_cache_list = curl_slist_append(NULL, redirect);
  if(!dns_cache_list) {
    fprintf(stderr, "curl_slist_append() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  easy_init(easy);

  easy_setopt(easy, CURLOPT_URL, URL);
  easy_setopt(easy, CURLOPT_HEADER, 1L);
  easy_setopt(easy, CURLOPT_RESOLVE, dns_cache_list);

  dup = curl_easy_duphandle(easy);
  if(dup) {
    curl_easy_cleanup(easy);
    easy = dup;
  }
  else {
    curl_slist_free_all(dns_cache_list);
    curl_easy_cleanup(easy);
    curl_global_cleanup();
    return CURLE_OUT_OF_MEMORY;
  }

  multi_init(multi);

  multi_add_handle(multi, easy);

  multi_perform(multi, &still_running);

  abort_on_test_timeout();

  while(still_running) {
    struct timeval timeout;
    fd_set fdread;
    fd_set fdwrite;
    fd_set fdexcep;
    int maxfd = -99;

    FD_ZERO(&fdread);
    FD_ZERO(&fdwrite);
    FD_ZERO(&fdexcep);
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    multi_fdset(multi, &fdread, &fdwrite, &fdexcep, &maxfd);

    /* At this point, maxfd is guaranteed to be greater or equal than -1. */

    select_test(maxfd + 1, &fdread, &fdwrite, &fdexcep, &timeout);

    abort_on_test_timeout();

    multi_perform(multi, &still_running);

    abort_on_test_timeout();
  }

test_cleanup:

#ifdef LIB1502
  /* undocumented cleanup sequence - type UA */
  curl_multi_cleanup(multi);
  curl_easy_cleanup(easy);
  curl_global_cleanup();
#endif

#ifdef LIB1503
  /* proper cleanup sequence - type PA */
  curl_multi_remove_handle(multi, easy);
  curl_multi_cleanup(multi);
  curl_easy_cleanup(easy);
  curl_global_cleanup();
#endif

#ifdef LIB1504
  /* undocumented cleanup sequence - type UB */
  curl_easy_cleanup(easy);
  curl_multi_cleanup(multi);
  curl_global_cleanup();
#endif

#ifdef LIB1505
  /* proper cleanup sequence - type PB */
  curl_multi_remove_handle(multi, easy);
  curl_easy_cleanup(easy);
  curl_multi_cleanup(multi);
  curl_global_cleanup();
#endif

  curl_slist_free_all(dns_cache_list);

  return res;
}
