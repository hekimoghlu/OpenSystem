/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 17, 2024.
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
#include "curlcheck.h"

#include "llist.h"

static CURL *easy;

static CURLcode unit_setup(void)
{
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);
  easy = curl_easy_init();
  if(!easy) {
    curl_global_cleanup();
    return CURLE_OUT_OF_MEMORY;
  }
  return res;
}

static void unit_stop(void)
{
  curl_easy_cleanup(easy);
  curl_global_cleanup();
}

UNITTEST_START
  int len;
  char *esc;

  esc = curl_easy_escape(easy, "", -1);
  fail_unless(esc == NULL, "negative string length can't work");

  esc = curl_easy_unescape(easy, "%41%41%41%41", -1, &len);
  fail_unless(esc == NULL, "negative string length can't work");

UNITTEST_STOP
