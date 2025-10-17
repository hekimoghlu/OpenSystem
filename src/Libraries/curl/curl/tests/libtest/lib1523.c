/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 26, 2024.
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

/* test case and code based on https://github.com/curl/curl/issues/3927 */

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

static int dload_progress_cb(void *a, curl_off_t b, curl_off_t c,
                             curl_off_t d, curl_off_t e)
{
  (void)a;
  (void)b;
  (void)c;
  (void)d;
  (void)e;
  return 0;
}

static size_t write_cb(char *d, size_t n, size_t l, void *p)
{
  /* take care of the data here, ignored in this example */
  (void)d;
  (void)p;
  return n*l;
}

static CURLcode run(CURL *hnd, long limit, long time)
{
  curl_easy_setopt(hnd, CURLOPT_LOW_SPEED_LIMIT, limit);
  curl_easy_setopt(hnd, CURLOPT_LOW_SPEED_TIME, time);
  return curl_easy_perform(hnd);
}

int test(char *URL)
{
  CURLcode ret;
  CURL *hnd;
  char buffer[CURL_ERROR_SIZE];
  curl_global_init(CURL_GLOBAL_ALL);
  hnd = curl_easy_init();
  curl_easy_setopt(hnd, CURLOPT_URL, URL);
  curl_easy_setopt(hnd, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(hnd, CURLOPT_ERRORBUFFER, buffer);
  curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 0L);
  curl_easy_setopt(hnd, CURLOPT_XFERINFOFUNCTION, dload_progress_cb);

  ret = run(hnd, 1, 2);
  if(ret)
    fprintf(stderr, "error %d: %s\n", ret, buffer);

  ret = run(hnd, 12000, 1);
  if(ret != CURLE_OPERATION_TIMEDOUT)
    fprintf(stderr, "error %d: %s\n", ret, buffer);
  else
    ret = CURLE_OK;

  curl_easy_cleanup(hnd);
  curl_global_cleanup();

  return (int)ret;
}
