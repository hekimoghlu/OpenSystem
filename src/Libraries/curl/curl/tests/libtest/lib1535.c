/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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

/* Test CURLINFO_PROTOCOL */

int test(char *URL)
{
  CURL *curl, *dupe = NULL;
  long protocol;
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  /* Test that protocol is properly initialized on curl_easy_init.
  */

  CURL_IGNORE_DEPRECATION(
    res = curl_easy_getinfo(curl, CURLINFO_PROTOCOL, &protocol);
  )
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(protocol) {
    fprintf(stderr, "%s:%d protocol init failed; expected 0 but is %ld\n",
            __FILE__, __LINE__, protocol);
    res = CURLE_FAILED_INIT;
    goto test_cleanup;
  }

  easy_setopt(curl, CURLOPT_URL, URL);

  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_perform() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }

  /* Test that a protocol is properly set after receiving an HTTP resource.
  */

  CURL_IGNORE_DEPRECATION(
    res = curl_easy_getinfo(curl, CURLINFO_PROTOCOL, &protocol);
  )
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(protocol != CURLPROTO_HTTP) {
    fprintf(stderr, "%s:%d protocol of http resource is incorrect; "
            "expected %d but is %ld\n",
            __FILE__, __LINE__, CURLPROTO_HTTP, protocol);
    res = CURLE_HTTP_RETURNED_ERROR;
    goto test_cleanup;
  }

  /* Test that a protocol is properly initialized on curl_easy_duphandle.
  */

  dupe = curl_easy_duphandle(curl);
  if(!dupe) {
    fprintf(stderr, "%s:%d curl_easy_duphandle() failed\n",
            __FILE__, __LINE__);
    res = CURLE_FAILED_INIT;
    goto test_cleanup;
  }

  CURL_IGNORE_DEPRECATION(
    res = curl_easy_getinfo(dupe, CURLINFO_PROTOCOL, &protocol);
  )
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(protocol) {
    fprintf(stderr, "%s:%d protocol init failed; expected 0 but is %ld\n",
            __FILE__, __LINE__, protocol);
    res = CURLE_FAILED_INIT;
    goto test_cleanup;
  }


  /* Test that a protocol is properly initialized on curl_easy_reset.
  */

  curl_easy_reset(curl);

  CURL_IGNORE_DEPRECATION(
    res = curl_easy_getinfo(curl, CURLINFO_PROTOCOL, &protocol);
  )
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(protocol) {
    fprintf(stderr, "%s:%d protocol init failed; expected 0 but is %ld\n",
            __FILE__, __LINE__, protocol);
    res = CURLE_FAILED_INIT;
    goto test_cleanup;
  }

test_cleanup:
  curl_easy_cleanup(curl);
  curl_easy_cleanup(dupe);
  curl_global_cleanup();
  return (int)res;
}
