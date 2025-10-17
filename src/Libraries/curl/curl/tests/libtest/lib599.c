/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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

static int progress_callback(void *clientp, double dltotal,
                             double dlnow, double ultotal, double ulnow)
{
  (void)clientp;
  (void)ulnow;
  (void)ultotal;

  if((dltotal > 0.0) && (dlnow > dltotal)) {
    /* this should not happen with test case 599 */
    printf("%.0f > %.0f !!\n", dltotal, dlnow);
    return -1;
  }

  return 0;
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  double content_length = 0.0;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  /* First set the URL that is about to receive our POST. */
  test_setopt(curl, CURLOPT_URL, URL);

  /* we want to use our own progress function */
  test_setopt(curl, CURLOPT_NOPROGRESS, 0L);
  CURL_IGNORE_DEPRECATION(
    test_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_callback);
  )

  /* get verbose debug output please */
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  /* follow redirects */
  test_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  /* include headers in the output */
  test_setopt(curl, CURLOPT_HEADER, 1L);

  /* Perform the request, res will get the return code */
  res = curl_easy_perform(curl);

  if(!res) {
    FILE *moo;
    CURL_IGNORE_DEPRECATION(
      res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD,
                            &content_length);
    )
    moo = fopen(libtest_arg2, "wb");
    if(moo) {
      fprintf(moo, "CL %.0f\n", content_length);
      fclose(moo);
    }
  }

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
