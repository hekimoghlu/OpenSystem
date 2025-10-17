/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 22, 2024.
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
#define WAKEUP_NUM 10

int test(char *URL)
{
  CURLM *multi = NULL;
  int numfds;
  int i;
  int res = 0;
  struct timeval time_before_wait, time_after_wait;

  (void)URL;

  start_test_timing();

  global_init(CURL_GLOBAL_ALL);

  multi_init(multi);

  /* no wakeup */

  time_before_wait = tutil_tvnow();
  multi_poll(multi, NULL, 0, 1000, &numfds);
  time_after_wait = tutil_tvnow();

  if(tutil_tvdiff(time_after_wait, time_before_wait) < 500) {
    fprintf(stderr, "%s:%d curl_multi_poll returned too early\n",
            __FILE__, __LINE__);
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  abort_on_test_timeout();

  /* try a single wakeup */

  res_multi_wakeup(multi);

  time_before_wait = tutil_tvnow();
  multi_poll(multi, NULL, 0, 1000, &numfds);
  time_after_wait = tutil_tvnow();

  if(tutil_tvdiff(time_after_wait, time_before_wait) > 500) {
    fprintf(stderr, "%s:%d curl_multi_poll returned too late\n",
            __FILE__, __LINE__);
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  abort_on_test_timeout();

  /* previous wakeup should not wake up this */

  time_before_wait = tutil_tvnow();
  multi_poll(multi, NULL, 0, 1000, &numfds);
  time_after_wait = tutil_tvnow();

  if(tutil_tvdiff(time_after_wait, time_before_wait) < 500) {
    fprintf(stderr, "%s:%d curl_multi_poll returned too early\n",
            __FILE__, __LINE__);
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  abort_on_test_timeout();

  /* try lots of wakeup */

  for(i = 0; i < WAKEUP_NUM; ++i)
    res_multi_wakeup(multi);

  time_before_wait = tutil_tvnow();
  multi_poll(multi, NULL, 0, 1000, &numfds);
  time_after_wait = tutil_tvnow();

  if(tutil_tvdiff(time_after_wait, time_before_wait) > 500) {
    fprintf(stderr, "%s:%d curl_multi_poll returned too late\n",
            __FILE__, __LINE__);
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  abort_on_test_timeout();

  /* Even lots of previous wakeups should not wake up this. */

  time_before_wait = tutil_tvnow();
  multi_poll(multi, NULL, 0, 1000, &numfds);
  time_after_wait = tutil_tvnow();

  if(tutil_tvdiff(time_after_wait, time_before_wait) < 500) {
    fprintf(stderr, "%s:%d curl_multi_poll returned too early\n",
            __FILE__, __LINE__);
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  abort_on_test_timeout();

test_cleanup:

  curl_multi_cleanup(multi);
  curl_global_cleanup();

  return res;
}
