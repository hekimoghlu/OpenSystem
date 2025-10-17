/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 3, 2024.
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
  CURLcode res = CURLE_OK;
  CURLSH *share;
  CURL *curl;

  curl_global_init(CURL_GLOBAL_ALL);

  share = curl_share_init();
  curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_COOKIE);

  curl = curl_easy_init();
  test_setopt(curl, CURLOPT_SHARE, share);

  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_HEADER, 1L);
  test_setopt(curl, CURLOPT_PROXY, URL);
  test_setopt(curl, CURLOPT_URL, "http://localhost/");

  test_setopt(curl, CURLOPT_COOKIEFILE, "");

  /* Set a cookie without Max-age or Expires */
  test_setopt(curl, CURLOPT_COOKIELIST, "Set-Cookie: c1=v1; domain=localhost");

  res = curl_easy_perform(curl);
  if(res) {
    fprintf(stderr, "curl_easy_perform() failed: %s\n",
            curl_easy_strerror(res));
  }

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);
    curl_share_cleanup(share);
  curl_global_cleanup();

  return (int)res;
}
