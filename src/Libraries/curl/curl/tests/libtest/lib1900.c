/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 7, 2022.
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

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

int test(char *URL)
{
  CURLcode res = CURLE_OK;
  CURL *hnd = NULL;
  CURL *second = NULL;

  global_init(CURL_GLOBAL_ALL);

  easy_init(hnd);
  easy_setopt(hnd, CURLOPT_URL, URL);
  easy_setopt(hnd, CURLOPT_HSTS, "first-hsts.txt");
  easy_setopt(hnd, CURLOPT_HSTS, "second-hsts.txt");

  second = curl_easy_duphandle(hnd);

  curl_easy_cleanup(hnd);
  curl_easy_cleanup(second);
  curl_global_cleanup();
  return 0;

test_cleanup:
  curl_easy_cleanup(hnd);
  curl_easy_cleanup(second);
  curl_global_cleanup();
  return (int)res;
}
