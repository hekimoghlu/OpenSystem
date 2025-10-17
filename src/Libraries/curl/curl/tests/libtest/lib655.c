/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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

static const char *TEST_DATA_STRING = "Test data";
static int cb_count = 0;

static int
resolver_alloc_cb_fail(void *resolver_state, void *reserved, void *userdata)
{
  (void)resolver_state;
  (void)reserved;

  cb_count++;
  if(strcmp(userdata, TEST_DATA_STRING)) {
    fprintf(stderr, "Invalid test data received");
    exit(1);
  }

  return 1;
}

static int
resolver_alloc_cb_pass(void *resolver_state, void *reserved, void *userdata)
{
  (void)resolver_state;
  (void)reserved;

  cb_count++;
  if(strcmp(userdata, TEST_DATA_STRING)) {
    fprintf(stderr, "Invalid test data received");
    exit(1);
  }

  return 0;
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }
  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  /* First set the URL that is about to receive our request. */
  test_setopt(curl, CURLOPT_URL, URL);

  test_setopt(curl, CURLOPT_RESOLVER_START_DATA, TEST_DATA_STRING);
  test_setopt(curl, CURLOPT_RESOLVER_START_FUNCTION, resolver_alloc_cb_fail);

  /* this should fail */
  res = curl_easy_perform(curl);
  if(res != CURLE_COULDNT_RESOLVE_HOST) {
    fprintf(stderr, "curl_easy_perform should have returned "
            "CURLE_COULDNT_RESOLVE_HOST but instead returned error %d\n", res);
    if(res == CURLE_OK)
      res = TEST_ERR_FAILURE;
    goto test_cleanup;
  }

  test_setopt(curl, CURLOPT_RESOLVER_START_FUNCTION, resolver_alloc_cb_pass);

  /* this should succeed */
  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "curl_easy_perform failed.\n");
    goto test_cleanup;
  }

  if(cb_count != 2) {
    fprintf(stderr, "Unexpected number of callbacks: %d\n", cb_count);
    res = TEST_ERR_FAILURE;
    goto test_cleanup;
  }

test_cleanup:
  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)res;
}
