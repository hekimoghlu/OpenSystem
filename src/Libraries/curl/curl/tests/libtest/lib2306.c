/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include "testtrace.h"

#include <curl/curl.h>

#define URL2 libtest_arg2

int test(char *URL)
{
  /* first a fine GET response, then a bad one */
  CURL *cl;
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);

  easy_init(cl);
  easy_setopt(cl, CURLOPT_URL, URL);
  easy_setopt(cl, CURLOPT_VERBOSE, 1L);
  res = curl_easy_perform(cl);
  if(res)
    goto test_cleanup;

  /* reuse handle, do a second transfer */
  easy_setopt(cl, CURLOPT_URL, URL2);
  res = curl_easy_perform(cl);

test_cleanup:
  curl_easy_cleanup(cl);
  curl_global_cleanup();
  return res;
}
