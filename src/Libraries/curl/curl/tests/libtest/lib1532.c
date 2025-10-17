/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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

/* Test CURLINFO_RESPONSE_CODE */

int test(char *URL)
{
  CURL *curl;
  long httpcode;
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  easy_setopt(curl, CURLOPT_URL, URL);

  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_perform() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }

  res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpcode);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(httpcode != 200) {
    fprintf(stderr, "%s:%d unexpected response code %ld\n",
            __FILE__, __LINE__, httpcode);
    res = CURLE_HTTP_RETURNED_ERROR;
    goto test_cleanup;
  }

  /* Test for a regression of github bug 1017 (response code does not reset) */
  curl_easy_reset(curl);

  res = curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpcode);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(httpcode) {
    fprintf(stderr, "%s:%d curl_easy_reset failed to zero the response code\n"
            "possible regression of github bug 1017\n", __FILE__, __LINE__);
    res = CURLE_HTTP_RETURNED_ERROR;
    goto test_cleanup;
  }

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();
  return (int)res;
}
