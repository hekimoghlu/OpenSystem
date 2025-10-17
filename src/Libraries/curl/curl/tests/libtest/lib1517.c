/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 21, 2025.
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

static char data[]="this is what we post to the silly web server\n";

struct WriteThis {
  char *readptr;
  size_t sizeleft;
};

static size_t read_callback(char *ptr, size_t size, size_t nmemb, void *userp)
{
  struct WriteThis *pooh = (struct WriteThis *)userp;
  size_t tocopy = size * nmemb;

  /* Wait one second before return POST data          *
   * so libcurl will wait before sending request body */
  wait_ms(1000);

  if(tocopy < 1 || !pooh->sizeleft)
    return 0;

  if(pooh->sizeleft < tocopy)
    tocopy = pooh->sizeleft;

  memcpy(ptr, pooh->readptr, tocopy);/* copy requested data */
  pooh->readptr += tocopy;           /* advance pointer */
  pooh->sizeleft -= tocopy;          /* less data left */
  return tocopy;
}

int test(char *URL)
{
  CURL *curl;
  CURLcode res = CURLE_OK;

  struct WriteThis pooh;

  if(!strcmp(URL, "check")) {
#if (defined(_WIN32) || defined(__CYGWIN__))
    printf("Windows TCP does not deliver response data but reports "
           "CONNABORTED\n");
    return 1; /* skip since test will fail on Windows without workaround */
#else
    return 0; /* sure, run this! */
#endif
  }

  pooh.readptr = data;
  pooh.sizeleft = strlen(data);

  if(curl_global_init(CURL_GLOBAL_ALL)) {
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

  /* Now specify we want to POST data */
  test_setopt(curl, CURLOPT_POST, 1L);

  /* Set the expected POST size */
  test_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)pooh.sizeleft);

  /* we want to use our own read function */
  test_setopt(curl, CURLOPT_READFUNCTION, read_callback);

  /* pointer to pass to our read function */
  test_setopt(curl, CURLOPT_READDATA, &pooh);

  /* get verbose debug output please */
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  /* include headers in the output */
  test_setopt(curl, CURLOPT_HEADER, 1L);

  /* detect HTTP error codes >= 400 */
  /* test_setopt(curl, CURLOPT_FAILONERROR, 1L); */


  /* Perform the request, res will get the return code */
  res = curl_easy_perform(curl);

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
