/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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
  CURLcode ret = CURLE_OK;
  CURL *curl = NULL;
  curl_off_t retry_after;
  char *follow_url = NULL;

  curl_global_init(CURL_GLOBAL_ALL);
  curl = curl_easy_init();

  if(curl) {
    curl_easy_setopt(curl, CURLOPT_URL, URL);
    ret = curl_easy_perform(curl);
    if(ret) {
      fprintf(stderr, "%s:%d curl_easy_perform() failed with code %d (%s)\n",
          __FILE__, __LINE__, ret, curl_easy_strerror(ret));
      goto test_cleanup;
    }
    curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &follow_url);
    curl_easy_getinfo(curl, CURLINFO_RETRY_AFTER, &retry_after);
    printf("Retry-After %" CURL_FORMAT_CURL_OFF_T "\n", retry_after);
    curl_easy_setopt(curl, CURLOPT_URL, follow_url);
    ret = curl_easy_perform(curl);
    if(ret) {
      fprintf(stderr, "%s:%d curl_easy_perform() failed with code %d (%s)\n",
          __FILE__, __LINE__, ret, curl_easy_strerror(ret));
      goto test_cleanup;
    }

    curl_easy_reset(curl);
    curl_easy_getinfo(curl, CURLINFO_RETRY_AFTER, &retry_after);
    printf("Retry-After %" CURL_FORMAT_CURL_OFF_T "\n", retry_after);
  }

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return ret;
}
