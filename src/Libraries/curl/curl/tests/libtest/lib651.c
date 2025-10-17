/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 4, 2024.
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
#define CURL_DISABLE_DEPRECATION  /* Using and testing the form api */
#include "test.h"

#include "memdebug.h"

static char buffer[17000]; /* more than 16K */

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;
  CURLFORMcode formrc;
  struct curl_httppost *formpost = NULL;
  struct curl_httppost *lastptr = NULL;

  /* create a buffer with AAAA...BBBBB...CCCC...etc */
  int i;
  int size = (int)sizeof(buffer)/1000;

  for(i = 0; i < size ; i++)
    memset(&buffer[i * 1000], 65 + i, 1000);

  buffer[ sizeof(buffer)-1] = 0; /* null-terminate */

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  /* Check proper name and data copying. */
  formrc = curl_formadd(&formpost, &lastptr,
                        CURLFORM_COPYNAME, "hello",
                        CURLFORM_COPYCONTENTS, buffer,
                        CURLFORM_END);

  if(formrc)
    printf("curl_formadd(1) = %d\n", (int) formrc);


  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_formfree(formpost);
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  /* First set the URL that is about to receive our POST. */
  test_setopt(curl, CURLOPT_URL, URL);

  /* send a multi-part formpost */
  test_setopt(curl, CURLOPT_HTTPPOST, formpost);

  /* get verbose debug output please */
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  /* include headers in the output */
  test_setopt(curl, CURLOPT_HEADER, 1L);

  /* Perform the request, res will get the return code */
  res = curl_easy_perform(curl);

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);

  /* now cleanup the formpost chain */
  curl_formfree(formpost);

  curl_global_cleanup();

  return res;
}
