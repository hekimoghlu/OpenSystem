/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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

#include "memdebug.h"

#define TEST_HANG_TIMEOUT (60 * 1000)

static int new_fnmatch(void *ptr,
                       const char *pattern, const char *string)
{
  (void)ptr;
  (void)pattern;
  (void)string;
  return CURL_FNMATCHFUNC_MATCH;
}

int test(char *URL)
{
  int res;
  CURL *curl;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  test_setopt(curl, CURLOPT_URL, URL);
  test_setopt(curl, CURLOPT_WILDCARDMATCH, 1L);
  test_setopt(curl, CURLOPT_FNMATCH_FUNCTION, new_fnmatch);
  test_setopt(curl, CURLOPT_TIMEOUT_MS, (long) TEST_HANG_TIMEOUT);

  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "curl_easy_perform() failed %d\n", res);
    goto test_cleanup;
  }
  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "curl_easy_perform() failed %d\n", res);
    goto test_cleanup;
  }

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();
  return res;
}
