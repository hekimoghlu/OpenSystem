/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

#define TEST_HANG_TIMEOUT 60 * 1000

/*
 * Source code in here hugely as reported in bug report 651460 by
 * Christopher R. Palmer.
 *
 * Use multi interface to get HTTPS document over proxy, and provide
 * auth info.
 */

int test(char *URL)
{
  CURL *c = NULL;
  CURLM *m = NULL;
  int res = 0;
  int running;

  start_test_timing();

  global_init(CURL_GLOBAL_ALL);

  easy_init(c);

  easy_setopt(c, CURLOPT_PROXY, libtest_arg2); /* set in first.c */
  easy_setopt(c, CURLOPT_URL, URL);
  easy_setopt(c, CURLOPT_USERPWD, "test:ing");
  easy_setopt(c, CURLOPT_PROXYUSERPWD, "test:ing");
  easy_setopt(c, CURLOPT_HTTPPROXYTUNNEL, 1L);
  easy_setopt(c, CURLOPT_HEADER, 1L);
  easy_setopt(c, CURLOPT_VERBOSE, 1L);

  multi_init(m);

  multi_add_handle(m, c);

  for(;;) {
    struct timeval interval;
    fd_set rd, wr, exc;
    int maxfd = -99;

    interval.tv_sec = 1;
    interval.tv_usec = 0;

    multi_perform(m, &running);

    abort_on_test_timeout();

    if(!running)
      break; /* done */

    FD_ZERO(&rd);
    FD_ZERO(&wr);
    FD_ZERO(&exc);

    multi_fdset(m, &rd, &wr, &exc, &maxfd);

    /* At this point, maxfd is guaranteed to be greater or equal than -1. */

    select_test(maxfd + 1, &rd, &wr, &exc, &interval);

    abort_on_test_timeout();
  }

test_cleanup:

  /* proper cleanup sequence - type PA */

  curl_multi_remove_handle(m, c);
  curl_multi_cleanup(m);
  curl_easy_cleanup(c);
  curl_global_cleanup();

  return res;
}
