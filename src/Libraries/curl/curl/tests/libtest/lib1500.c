/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 9, 2024.
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

int test(char *URL)
{
  CURL *curls = NULL;
  CURLM *multi = NULL;
  int still_running;
  int i = TEST_ERR_FAILURE;
  int res = 0;
  CURLMsg *msg;

  start_test_timing();

  global_init(CURL_GLOBAL_ALL);

  multi_init(multi);

  easy_init(curls);

  easy_setopt(curls, CURLOPT_URL, URL);
  easy_setopt(curls, CURLOPT_HEADER, 1L);

  multi_add_handle(multi, curls);

  multi_perform(multi, &still_running);

  abort_on_test_timeout();

  while(still_running) {
    int num;
    res = curl_multi_wait(multi, NULL, 0, TEST_HANG_TIMEOUT, &num);
    if(res != CURLM_OK) {
      printf("curl_multi_wait() returned %d\n", res);
      res = TEST_ERR_MAJOR_BAD;
      goto test_cleanup;
    }

    abort_on_test_timeout();

    multi_perform(multi, &still_running);

    abort_on_test_timeout();
  }

  msg = curl_multi_info_read(multi, &still_running);
  if(msg)
    /* this should now contain a result code from the easy handle,
       get it */
    i = msg->data.result;

test_cleanup:

  /* undocumented cleanup sequence - type UA */

  curl_multi_cleanup(multi);
  curl_easy_cleanup(curls);
  curl_global_cleanup();

  if(res)
    i = res;

  return i; /* return the final return code */
}
