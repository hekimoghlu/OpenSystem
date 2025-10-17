/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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

/* Test case code based on source in a bug report filed by James Bursa on
   28 Apr 2004 */

int test(char *URL)
{
  CURLcode code;
  int rc = 99;

  code = curl_global_init(CURL_GLOBAL_ALL);
  if(code == CURLE_OK) {
    CURL *curl = curl_easy_init();
    if(curl) {
      CURL *curl2;

      curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
      curl_easy_setopt(curl, CURLOPT_HEADER, 1L);

      curl2 = curl_easy_duphandle(curl);
      if(curl2) {

        code = curl_easy_setopt(curl2, CURLOPT_URL, URL);
        if(code == CURLE_OK) {

          code = curl_easy_perform(curl2);
          if(code == CURLE_OK)
            rc = 0;
          else
            rc = 1;
        }
        else
          rc = 2;

        curl_easy_cleanup(curl2);
      }
      else
        rc = 3;

      curl_easy_cleanup(curl);
    }
    else
      rc = 4;

    curl_global_cleanup();
  }
  else
    rc = 5;

  return rc;
}
