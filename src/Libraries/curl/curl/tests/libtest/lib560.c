/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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
 * Simply download an HTTPS file!
 *
 * This test was added after the HTTPS-using-multi-interface with OpenSSL
 * regression of 7.19.1 to hopefully prevent this embarrassing mistake from
 * appearing again... Unfortunately the bug wasn't triggered by this test,
 * which presumably is because the connect to a local server is too
 * fast/different compared to the real/distant servers we saw the bug happen
 * with.
 */
int test(char *URL)
{
  CURL *http_handle = NULL;
  CURLM *multi_handle = NULL;
  int res = 0;

  int still_running; /* keep number of running handles */

  start_test_timing();

  /*
  ** curl_global_init called indirectly from curl_easy_init.
  */

  easy_init(http_handle);

  /* set options */
  easy_setopt(http_handle, CURLOPT_URL, URL);
  easy_setopt(http_handle, CURLOPT_HEADER, 1L);
  easy_setopt(http_handle, CURLOPT_SSL_VERIFYPEER, 0L);
  easy_setopt(http_handle, CURLOPT_SSL_VERIFYHOST, 0L);

  /* init a multi stack */
  multi_init(multi_handle);

  /* add the individual transfers */
  multi_add_handle(multi_handle, http_handle);

  /* we start some action by calling perform right away */
  multi_perform(multi_handle, &still_running);

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

    /* set a suitable timeout to play around with */
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    /* get file descriptors from the transfers */
    multi_fdset(multi_handle, &fdread, &fdwrite, &fdexcep, &maxfd);

    /* At this point, maxfd is guaranteed to be greater or equal than -1. */

    select_test(maxfd + 1, &fdread, &fdwrite, &fdexcep, &timeout);

    abort_on_test_timeout();

    /* timeout or readable/writable sockets */
    multi_perform(multi_handle, &still_running);

    abort_on_test_timeout();
  }

test_cleanup:

  /* undocumented cleanup sequence - type UA */

  curl_multi_cleanup(multi_handle);
  curl_easy_cleanup(http_handle);
  curl_global_cleanup();

  return res;
}
