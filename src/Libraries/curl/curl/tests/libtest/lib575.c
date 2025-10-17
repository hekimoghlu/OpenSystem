/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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

#include <fcntl.h>

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

#define TEST_HANG_TIMEOUT 60 * 1000

/* 3x download!
 * 1. normal
 * 2. dup handle
 * 3. with multi interface
 */

int test(char *URL)
{
  CURL *handle = NULL;
  CURL *duphandle = NULL;
  CURLM *mhandle = NULL;
  int res = 0;
  int still_running = 0;

  start_test_timing();

  global_init(CURL_GLOBAL_ALL);

  easy_init(handle);

  easy_setopt(handle, CURLOPT_URL, URL);
  easy_setopt(handle, CURLOPT_WILDCARDMATCH, 1L);
  easy_setopt(handle, CURLOPT_VERBOSE, 1L);

  res = curl_easy_perform(handle);
  if(res)
    goto test_cleanup;

  res = curl_easy_perform(handle);
  if(res)
    goto test_cleanup;

  duphandle = curl_easy_duphandle(handle);
  if(!duphandle)
    goto test_cleanup;
  curl_easy_cleanup(handle);
  handle = duphandle;

  multi_init(mhandle);

  multi_add_handle(mhandle, handle);

  multi_perform(mhandle, &still_running);

  abort_on_test_timeout();

  while(still_running) {
    struct timeval timeout;
    fd_set fdread;
    fd_set fdwrite;
    fd_set fdexcep;
    int maxfd = -99;

    timeout.tv_sec = 0;
    timeout.tv_usec = 100000L; /* 100 ms */

    FD_ZERO(&fdread);
    FD_ZERO(&fdwrite);
    FD_ZERO(&fdexcep);

    multi_fdset(mhandle, &fdread, &fdwrite, &fdexcep, &maxfd);

    /* At this point, maxfd is guaranteed to be greater or equal than -1. */

    select_test(maxfd + 1, &fdread, &fdwrite, &fdexcep, &timeout);

    abort_on_test_timeout();

    multi_perform(mhandle, &still_running);

    abort_on_test_timeout();
  }

test_cleanup:

  /* undocumented cleanup sequence - type UA */

  curl_multi_cleanup(mhandle);
  curl_easy_cleanup(handle);
  curl_global_cleanup();

  return res;
}
