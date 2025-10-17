/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 4, 2025.
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

static int xferinfo(void *p,
                    curl_off_t dltotal, curl_off_t dlnow,
                    curl_off_t ultotal, curl_off_t ulnow)
{
  (void)p;
  (void)dlnow;
  (void)dltotal;
  (void)ulnow;
  (void)ultotal;
  fprintf(stderr, "xferinfo fail!\n");
  return 1; /* fail as fast as we can */
}

int test(char *URL)
{
  CURL *curls = NULL;
  CURLM *multi = NULL;
  int still_running;
  int i = 0;
  int res = 0;
  curl_mimepart *field = NULL;
  curl_mime *mime = NULL;
  int counter = 1;

  start_test_timing();

  global_init(CURL_GLOBAL_ALL);

  multi_init(multi);

  easy_init(curls);

  mime = curl_mime_init(curls);
  field = curl_mime_addpart(mime);
  curl_mime_name(field, "name");
  curl_mime_data(field, "value", CURL_ZERO_TERMINATED);

  easy_setopt(curls, CURLOPT_URL, URL);
  easy_setopt(curls, CURLOPT_HEADER, 1L);
  easy_setopt(curls, CURLOPT_VERBOSE, 1L);
  easy_setopt(curls, CURLOPT_MIMEPOST, mime);
  easy_setopt(curls, CURLOPT_USERPWD, "u:s");
  easy_setopt(curls, CURLOPT_XFERINFOFUNCTION, xferinfo);
  easy_setopt(curls, CURLOPT_NOPROGRESS, 1L);

  multi_add_handle(multi, curls);

  multi_perform(multi, &still_running);

  abort_on_test_timeout();

  while(still_running && counter--) {
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

test_cleanup:

  curl_mime_free(mime);
  curl_multi_remove_handle(multi, curls);
  curl_multi_cleanup(multi);
  curl_easy_cleanup(curls);
  curl_global_cleanup();

  if(res)
    i = res;

  return i; /* return the final return code */
}
