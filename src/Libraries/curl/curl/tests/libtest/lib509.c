/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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

#include <string.h>

/*
 * This test uses these funny custom memory callbacks for the only purpose
 * of verifying that curl_global_init_mem() functionality is present in
 * libcurl and that it works unconditionally no matter how libcurl is built,
 * nothing more.
 *
 * Do not include memdebug.h in this source file, and do not use directly
 * memory related functions in this file except those used inside custom
 * memory callbacks which should be calling 'the real thing'.
 */

static int seen;

static void *custom_calloc(size_t nmemb, size_t size)
{
  seen++;
  return (calloc)(nmemb, size);
}

static void *custom_malloc(size_t size)
{
  seen++;
  return (malloc)(size);
}

static char *custom_strdup(const char *ptr)
{
  seen++;
  return (strdup)(ptr);
}

static void *custom_realloc(void *ptr, size_t size)
{
  seen++;
  return (realloc)(ptr, size);
}

static void custom_free(void *ptr)
{
  seen++;
  (free)(ptr);
}


int test(char *URL)
{
  unsigned char a[] = {0x2f, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
                       0x91, 0xa2, 0xb3, 0xc4, 0xd5, 0xe6, 0xf7};
  CURLcode res;
  CURL *curl;
  int asize;
  char *str = NULL;
  (void)URL;

  res = curl_global_init_mem(CURL_GLOBAL_ALL,
                             custom_malloc,
                             custom_free,
                             custom_realloc,
                             custom_strdup,
                             custom_calloc);
  if(res != CURLE_OK) {
    fprintf(stderr, "curl_global_init_mem() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  test_setopt(curl, CURLOPT_USERAGENT, "test509"); /* uses strdup() */

  asize = (int)sizeof(a);
  str = curl_easy_escape(curl, (char *)a, asize); /* uses realloc() */

  if(seen)
    printf("Callbacks were invoked!\n");

test_cleanup:

  if(str)
    curl_free(str);

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)res;
}
