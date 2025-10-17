/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#include "memdebug.h"

/*
 * FTP get with NOBODY but no HEADER
 */

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  /* get a curl handle */
  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  /* enable verbose */
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  /* enable NOBODY */
  test_setopt(curl, CURLOPT_NOBODY, 1L);

  /* disable HEADER */
  test_setopt(curl, CURLOPT_HEADER, 0L);

  /* specify target */
  test_setopt(curl, CURLOPT_URL, URL);

  /* Now run off and do what you've been told! */
  res = curl_easy_perform(curl);

test_cleanup:

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
