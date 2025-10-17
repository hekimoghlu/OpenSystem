/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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
 * Test case converted from bug report #1318 by Petr Novak.
 *
 * Before the fix, this test program returned 52 (CURLE_GOT_NOTHING) instead
 * of 42 (CURLE_ABORTED_BY_CALLBACK).
 */

#include "test.h"

#include "memdebug.h"

static int progressKiller(void *arg,
                          double dltotal,
                          double dlnow,
                          double ultotal,
                          double ulnow)
{
  (void)arg;
  (void)dltotal;
  (void)dlnow;
  (void)ultotal;
  (void)ulnow;
  printf("PROGRESSFUNCTION called\n");
  return 1;
}

int test(char *URL)
{
  CURL *curl;
  int res = 0;

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  easy_setopt(curl, CURLOPT_URL, URL);
  easy_setopt(curl, CURLOPT_TIMEOUT, (long)7);
  easy_setopt(curl, CURLOPT_NOSIGNAL, (long)1);
  CURL_IGNORE_DEPRECATION(
    easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progressKiller);
    easy_setopt(curl, CURLOPT_PROGRESSDATA, NULL);
  )
  easy_setopt(curl, CURLOPT_NOPROGRESS, (long)0);

  res = curl_easy_perform(curl);

test_cleanup:

  /* undocumented cleanup sequence - type UA */

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
