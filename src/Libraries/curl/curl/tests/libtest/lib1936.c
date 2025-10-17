/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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

int test(char *URL)
{
  CURL *curl;
  CURLcode res = TEST_ERR_MAJOR_BAD;
  struct curl_slist *connect_to = NULL;
  struct curl_slist *list = NULL;

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

  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_AWS_SIGV4, "xxx:yyy:rrr:sss");
  test_setopt(curl, CURLOPT_USERPWD, "xxx:yyy");
  test_setopt(curl, CURLOPT_HEADER, 0L);
  test_setopt(curl, CURLOPT_URL, URL);
  if(libtest_arg2) {
    connect_to = curl_slist_append(connect_to, libtest_arg2);
  }
  test_setopt(curl, CURLOPT_CONNECT_TO, connect_to);
  list = curl_slist_append(list, "Content-Type: application/json");
  test_setopt(curl, CURLOPT_HTTPHEADER, list);

  res = curl_easy_perform(curl);

test_cleanup:

  curl_slist_free_all(connect_to);
  curl_slist_free_all(list);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
