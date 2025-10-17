/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 4, 2022.
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
/* Based on Alex Fishman's bug report on September 30, 2007 */

#include "test.h"

#include "memdebug.h"

int test(char *URL)
{
  static const unsigned char a[] = {
      0x9c, 0x26, 0x4b, 0x3d, 0x49, 0x4, 0xa1, 0x1,
      0xe0, 0xd8, 0x7c,  0x20, 0xb7, 0xef, 0x53, 0x29, 0xfa,
      0x1d, 0x57, 0xe1};

  CURL *easy;
  CURLcode res = CURLE_OK;
  (void)URL;

  global_init(CURL_GLOBAL_ALL);
  easy = curl_easy_init();
  if(!easy) {
    fprintf(stderr, "curl_easy_init() failed\n");
    res = TEST_ERR_MAJOR_BAD;
  }
  else {
    int asize = (int)sizeof(a);
    char *s = curl_easy_escape(easy, (const char *)a, asize);

    if(s) {
      printf("%s\n", s);
      curl_free(s);
    }

    s = curl_easy_escape(easy, "", 0);
    if(s) {
      printf("IN: '' OUT: '%s'\n", s);
      curl_free(s);
    }
    s = curl_easy_escape(easy, " 123", 3);
    if(s) {
      printf("IN: ' 12' OUT: '%s'\n", s);
      curl_free(s);
    }

    curl_easy_cleanup(easy);
  }
  curl_global_cleanup();

  return (int)res;
}
