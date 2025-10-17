/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
  CURLcode ret = CURLE_OK;
  CURL *hnd;
  start_test_timing();

  curl_global_init(CURL_GLOBAL_ALL);

  hnd = curl_easy_init();
  if(hnd) {
    curl_easy_setopt(hnd, CURLOPT_URL, URL);
    curl_easy_setopt(hnd, CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(hnd, CURLOPT_ALTSVC, libtest_arg2);
    ret = curl_easy_perform(hnd);

    if(!ret) {
      /* make a copy and check that this also has alt-svc activated */
      CURL *also = curl_easy_duphandle(hnd);
      if(also) {
        ret = curl_easy_perform(also);
        /* we close the second handle first, which makes it store the alt-svc
           file only to get overwritten when the next handle is closed! */
        curl_easy_cleanup(also);
      }
    }

    curl_easy_reset(hnd);

    /* using the same file name for the alt-svc cache, this clobbers the
       content just written from the 'also' handle */
    curl_easy_cleanup(hnd);
  }
  curl_global_cleanup();
  return (int)ret;
}

