/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
/* argv1 = URL
 * argv2 = proxy
 * argv3 = proxyuser:password
 */

#include "test.h"

#include "memdebug.h"

#define UPLOADTHIS "this is the blurb we want to upload\n"

#ifndef LIB548
static size_t readcallback(char  *ptr,
                           size_t size,
                           size_t nmemb,
                           void *clientp)
{
  int *counter = (int *)clientp;

  if(*counter) {
    /* only do this once and then require a clearing of this */
    fprintf(stderr, "READ ALREADY DONE!\n");
    return 0;
  }
  (*counter)++; /* bump */

  if(size * nmemb >= strlen(UPLOADTHIS)) {
    fprintf(stderr, "READ!\n");
    strcpy(ptr, UPLOADTHIS);
    return strlen(UPLOADTHIS);
  }
  fprintf(stderr, "READ NOT FINE!\n");
  return 0;
}
static curlioerr ioctlcallback(CURL *handle,
                               int cmd,
                               void *clientp)
{
  int *counter = (int *)clientp;
  (void)handle; /* unused */
  if(cmd == CURLIOCMD_RESTARTREAD) {
    fprintf(stderr, "REWIND!\n");
    *counter = 0; /* clear counter to make the read callback restart */
  }
  return CURLIOE_OK;
}



#endif

int test(char *URL)
{
  CURLcode res;
  CURL *curl;
#ifndef LIB548
  int counter = 0;
#endif

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

  test_setopt(curl, CURLOPT_URL, URL);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);
  test_setopt(curl, CURLOPT_HEADER, 1L);
#ifdef LIB548
  /* set the data to POST with a mere pointer to a null-terminated string */
  test_setopt(curl, CURLOPT_POSTFIELDS, UPLOADTHIS);
#else
  /* 547 style, which means reading the POST data from a callback */
  CURL_IGNORE_DEPRECATION(
    test_setopt(curl, CURLOPT_IOCTLFUNCTION, ioctlcallback);
    test_setopt(curl, CURLOPT_IOCTLDATA, &counter);
  )
  test_setopt(curl, CURLOPT_READFUNCTION, readcallback);
  test_setopt(curl, CURLOPT_READDATA, &counter);
  /* We CANNOT do the POST fine without setting the size (or choose
     chunked)! */
  test_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(UPLOADTHIS));
#endif
  test_setopt(curl, CURLOPT_POST, 1L);
  test_setopt(curl, CURLOPT_PROXY, libtest_arg2);
  test_setopt(curl, CURLOPT_PROXYUSERPWD, libtest_arg3);
  test_setopt(curl, CURLOPT_PROXYAUTH,
                   (long) (CURLAUTH_NTLM | CURLAUTH_DIGEST | CURLAUTH_BASIC) );

  res = curl_easy_perform(curl);

test_cleanup:

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)res;
}
