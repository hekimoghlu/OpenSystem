/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 13, 2022.
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
  const unsigned char a[] = {0x2f, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
                             0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7};
  CURLcode res = CURLE_OK;
  char *ptr = NULL;
  int asize;
  int outlen = 0;
  char *raw;

  (void)URL; /* we don't use this */

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  asize = (int)sizeof(a);
  ptr = curl_easy_escape(NULL, (char *)a, asize);
  printf("%s\n", ptr);
  curl_free(ptr);

  /* deprecated API */
  ptr = curl_escape((char *)a, asize);
  if(!ptr) {
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  printf("%s\n", ptr);

  raw = curl_easy_unescape(NULL, ptr, (int)strlen(ptr), &outlen);
  printf("outlen == %d\n", outlen);
  printf("unescape == original? %s\n",
         memcmp(raw, a, outlen) ? "no" : "YES");
  curl_free(raw);

  /* deprecated API */
  raw = curl_unescape(ptr, (int)strlen(ptr));
  if(!raw) {
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }
  outlen = (int)strlen(raw);
  printf("[old] outlen == %d\n", outlen);
  printf("[old] unescape == original? %s\n",
         memcmp(raw, a, outlen) ? "no" : "YES");
  curl_free(raw);
  curl_free(ptr);

  /* weird input length */
  ptr = curl_easy_escape(NULL, (char *)a, -1);
  printf("escape -1 length: %s\n", ptr);

  /* weird input length */
  outlen = 2017; /* just a value */
  ptr = curl_easy_unescape(NULL, (char *)"moahahaha", -1, &outlen);
  printf("unescape -1 length: %s %d\n", ptr, outlen);

test_cleanup:
  curl_free(ptr);
  curl_global_cleanup();

  return (int)res;
}
