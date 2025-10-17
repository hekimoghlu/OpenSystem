/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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

/* Test CURLINFO_SCHEME */

int test(char *URL)
{
  CURL *curl, *dupe = NULL;
  char *scheme;
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  /* Test that scheme is properly initialized on curl_easy_init.
  */

  res = curl_easy_getinfo(curl, CURLINFO_SCHEME, &scheme);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(scheme) {
    fprintf(stderr, "%s:%d scheme init failed; expected NULL\n",
            __FILE__, __LINE__);
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

  /* Test that a scheme is properly set after receiving an HTTP resource.
  */

  res = curl_easy_getinfo(curl, CURLINFO_SCHEME, &scheme);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(!scheme || memcmp(scheme, "HTTP", 5) != 0) {
    fprintf(stderr, "%s:%d scheme of http resource is incorrect; "
            "expected 'HTTP' but is %s\n",
            __FILE__, __LINE__,
            (scheme == NULL ? "NULL" : "invalid"));
    res = CURLE_HTTP_RETURNED_ERROR;
    goto test_cleanup;
  }

  /* Test that a scheme is properly initialized on curl_easy_duphandle.
  */

  dupe = curl_easy_duphandle(curl);
  if(!dupe) {
    fprintf(stderr, "%s:%d curl_easy_duphandle() failed\n",
            __FILE__, __LINE__);
    res = CURLE_FAILED_INIT;
    goto test_cleanup;
  }

  res = curl_easy_getinfo(dupe, CURLINFO_SCHEME, &scheme);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(scheme) {
    fprintf(stderr, "%s:%d scheme init failed; expected NULL\n",
            __FILE__, __LINE__);
    res = CURLE_FAILED_INIT;
    goto test_cleanup;
  }


  /* Test that a scheme is properly initialized on curl_easy_reset.
  */

  curl_easy_reset(curl);

  res = curl_easy_getinfo(curl, CURLINFO_SCHEME, &scheme);
  if(res) {
    fprintf(stderr, "%s:%d curl_easy_getinfo() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }
  if(scheme) {
    fprintf(stderr, "%s:%d scheme init failed; expected NULL\n",
            __FILE__, __LINE__);
    res = CURLE_FAILED_INIT;
    goto test_cleanup;
  }

test_cleanup:
  curl_easy_cleanup(curl);
  curl_easy_cleanup(dupe);
  curl_global_cleanup();
  return (int)res;
}
