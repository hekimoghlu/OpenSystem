/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 19, 2024.
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
#include "memdebug.h"

int test(char *URL)
{
  CURLcode ret;
  CURL *hnd;
  curl_global_init(CURL_GLOBAL_ALL);

  hnd = curl_easy_init();
  curl_easy_setopt(hnd, CURLOPT_URL, URL);
  curl_easy_setopt(hnd, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(hnd, CURLOPT_HEADER, 1L);
  curl_easy_setopt(hnd, CURLOPT_USERPWD, "testuser:testpass");
  curl_easy_setopt(hnd, CURLOPT_USERAGENT, "lib1568");
  curl_easy_setopt(hnd, CURLOPT_HTTPAUTH, (long)CURLAUTH_DIGEST);
  curl_easy_setopt(hnd, CURLOPT_MAXREDIRS, 50L);
  curl_easy_setopt(hnd, CURLOPT_PORT, strtol(libtest_arg2, NULL, 10));

  ret = curl_easy_perform(hnd);

  curl_easy_cleanup(hnd);
  hnd = NULL;

  curl_global_cleanup();
  return (int)ret;
}
