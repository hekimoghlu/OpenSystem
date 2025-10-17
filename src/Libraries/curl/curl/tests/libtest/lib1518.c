/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

/* Test inspired by github issue 3340 */

static size_t writecb(char *buffer, size_t size, size_t nitems,
                      void *outstream)
{
  (void)buffer;
  (void)size;
  (void)nitems;
  (void)outstream;
  return 0;
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  long curlResponseCode;
  long curlRedirectCount;
  char *effectiveUrl = NULL;
  char *redirectUrl = NULL;
#ifdef LIB1543
  CURLU *urlu = NULL;
#endif
  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }
#ifdef LIB1543
  /* set CURLOPT_URLU */
  {
    CURLUcode rc = CURLUE_OK;
    urlu = curl_url();
    if(urlu)
      rc = curl_url_set(urlu, CURLUPART_URL, URL, CURLU_ALLOW_SPACE);
    if(!urlu || rc) {
      goto test_cleanup;
    }
    test_setopt(curl, CURLOPT_CURLU, urlu);
  }
  test_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
#else
  test_setopt(curl, CURLOPT_URL, URL);
  /* just to make it explicit and visible in this test: */
  test_setopt(curl, CURLOPT_FOLLOWLOCATION, 0L);
#endif


  /* Perform the request, res will get the return code */
  res = curl_easy_perform(curl);

  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &curlResponseCode);
  curl_easy_getinfo(curl, CURLINFO_REDIRECT_COUNT, &curlRedirectCount);
  curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &effectiveUrl);
  curl_easy_getinfo(curl, CURLINFO_REDIRECT_URL, &redirectUrl);
  test_setopt(curl, CURLOPT_WRITEFUNCTION, writecb);

  printf("res %d\n"
         "status %d\n"
         "redirects %d\n"
         "effectiveurl %s\n"
         "redirecturl %s\n",
         (int)res,
         (int)curlResponseCode,
         (int)curlRedirectCount,
         effectiveUrl,
         redirectUrl ? redirectUrl : "blank");

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_global_cleanup();
#ifdef LIB1543
  curl_url_cleanup(urlu);
#endif
  return res;
}
