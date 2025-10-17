/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 22, 2022.
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
  struct curl_slist *list = NULL;
  struct curl_slist *connect_to = NULL;

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
  test_setopt(curl, CURLOPT_AWS_SIGV4, "xxx");
  test_setopt(curl, CURLOPT_USERPWD, "xxx");
  test_setopt(curl, CURLOPT_HEADER, 0L);
  test_setopt(curl, CURLOPT_URL, URL);
  list = curl_slist_append(list, "test3: 1234");
  if(!list)
    goto test_cleanup;
  if(libtest_arg2) {
    connect_to = curl_slist_append(connect_to, libtest_arg2);
  }
  test_setopt(curl, CURLOPT_CONNECT_TO, connect_to);
  curl_slist_append(list, "Content-Type: application/json");

  /* 'name;' user headers with no value are used to send an empty header in the
     format 'name:' (note the semi-colon becomes a colon). this entry should
     show in SignedHeaders without an additional semi-colon, as any other
     header would. eg 'foo;test2;test3' and not 'foo;test2;;test3'. */
  curl_slist_append(list, "test2;");

  /* 'name:' user headers with no value are used to signal an internal header
     of that name should be removed and are not sent as a header. this entry
     should not show in SignedHeaders. */
  curl_slist_append(list, "test1:");

  /* 'name' user headers with no separator or value are invalid and ignored.
     this entry should not show in SignedHeaders. */
  curl_slist_append(list, "test0");

  curl_slist_append(list, "test_space: t\ts  m\t   end    ");
  curl_slist_append(list, "tesMixCase: MixCase");
  test_setopt(curl, CURLOPT_HTTPHEADER, list);

  res = curl_easy_perform(curl);

test_cleanup:

  curl_slist_free_all(connect_to);
  curl_slist_free_all(list);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
