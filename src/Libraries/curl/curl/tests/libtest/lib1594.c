/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 15, 2024.
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
/* Testing Retry-After header parser */

#include "test.h"

#include "memdebug.h"

int test(char *URL)
{
  struct curl_slist *header = NULL;
  curl_off_t retry;
  CURL *curl = NULL;
  int res = 0;

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  easy_setopt(curl, CURLOPT_URL, URL);

  res = curl_easy_perform(curl);
  if(res)
    goto test_cleanup;

  res = curl_easy_getinfo(curl, CURLINFO_RETRY_AFTER, &retry);
  if(res)
    goto test_cleanup;

#ifdef LIB1596
  /* we get a relative number of seconds, so add the number of seconds
     we're at to make it a somewhat stable number. Then remove accuracy. */
  retry += time(NULL);
  retry /= 10000;
#endif
  printf("Retry-After %" CURL_FORMAT_CURL_OFF_T "\n", retry);

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_slist_free_all(header);
  curl_global_cleanup();

  return res;
}
