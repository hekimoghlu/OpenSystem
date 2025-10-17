/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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

#include "testtrace.h"
#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

#define TEST_HANG_TIMEOUT 60 * 1000

/*
 * Get a single URL without select().
 */

int test(char *URL)
{
  CURL *c = NULL;
  CURLM *m = NULL;
  int res = 0;
  int running = 1;
  double connect_time = 0.0;
  double dbl_epsilon;

  dbl_epsilon = 1.0;
  do {
    dbl_epsilon /= 2.0;
  } while((double)(1.0 + (dbl_epsilon/2.0)) > (double)1.0);

  start_test_timing();

  global_init(CURL_GLOBAL_ALL);

  easy_init(c);

  easy_setopt(c, CURLOPT_HEADER, 1L);
  easy_setopt(c, CURLOPT_URL, URL);

  libtest_debug_config.nohex = 1;
  libtest_debug_config.tracetime = 1;
  easy_setopt(c, CURLOPT_DEBUGDATA, &libtest_debug_config);
  easy_setopt(c, CURLOPT_DEBUGFUNCTION, libtest_debug_cb);
  easy_setopt(c, CURLOPT_VERBOSE, 1L);

  multi_init(m);

  multi_add_handle(m, c);

  while(running) {
    struct timeval timeout;
    fd_set fdread, fdwrite, fdexcep;
    int maxfd = -99;

    timeout.tv_sec = 0;
    timeout.tv_usec = 100000L; /* 100 ms */

    multi_perform(m, &running);

    abort_on_test_timeout();

    if(!running)
      break; /* done */

    FD_ZERO(&fdread);
    FD_ZERO(&fdwrite);
    FD_ZERO(&fdexcep);

    multi_fdset(m, &fdread, &fdwrite, &fdexcep, &maxfd);

    /* At this point, maxfd is guaranteed to be greater or equal than -1. */

    select_test(maxfd + 1, &fdread, &fdwrite, &fdexcep, &timeout);

    abort_on_test_timeout();
  }

  curl_easy_getinfo(c, CURLINFO_CONNECT_TIME, &connect_time);
  if(connect_time < dbl_epsilon) {
    fprintf(stderr, "connect time %e is < epsilon %e\n",
            connect_time, dbl_epsilon);
    res = TEST_ERR_MAJOR_BAD;
  }

test_cleanup:

  /* proper cleanup sequence - type PA */

  curl_multi_remove_handle(m, c);
  curl_multi_cleanup(m);
  curl_easy_cleanup(c);
  curl_global_cleanup();

  return res;
}
