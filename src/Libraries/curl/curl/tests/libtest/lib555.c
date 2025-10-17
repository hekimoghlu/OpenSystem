/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
/* This test case is supposed to be identical to 547 except that this uses the
 * multi interface and 547 is easy interface.
 *
 * argv1 = URL
 * argv2 = proxy
 * argv3 = proxyuser:password
 */

#include "test.h"
#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

#define TEST_HANG_TIMEOUT 60 * 1000

static const char uploadthis[] =
  "this is the blurb we want to upload\n";

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

  if(size * nmemb >= strlen(uploadthis)) {
    fprintf(stderr, "READ!\n");
    strcpy(ptr, uploadthis);
    return strlen(uploadthis);
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


int test(char *URL)
{
  int res = 0;
  CURL *curl = NULL;
  int counter = 0;
  CURLM *m = NULL;
  int running = 1;

  start_test_timing();

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  easy_setopt(curl, CURLOPT_URL, URL);
  easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  easy_setopt(curl, CURLOPT_HEADER, 1L);

  /* read the POST data from a callback */
  CURL_IGNORE_DEPRECATION(
    easy_setopt(curl, CURLOPT_IOCTLFUNCTION, ioctlcallback);
    easy_setopt(curl, CURLOPT_IOCTLDATA, &counter);
  )
  easy_setopt(curl, CURLOPT_READFUNCTION, readcallback);
  easy_setopt(curl, CURLOPT_READDATA, &counter);
  /* We CANNOT do the POST fine without setting the size (or choose
     chunked)! */
  easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)strlen(uploadthis));

  easy_setopt(curl, CURLOPT_POST, 1L);
  easy_setopt(curl, CURLOPT_PROXY, libtest_arg2);
  easy_setopt(curl, CURLOPT_PROXYUSERPWD, libtest_arg3);
  easy_setopt(curl, CURLOPT_PROXYAUTH,
                   (long) (CURLAUTH_NTLM | CURLAUTH_DIGEST | CURLAUTH_BASIC) );

  multi_init(m);

  multi_add_handle(m, curl);

  while(running) {
    struct timeval timeout;
    fd_set fdread, fdwrite, fdexcep;
    int maxfd = -99;

    timeout.tv_sec = 0;
    timeout.tv_usec = 100000L; /* 100 ms */

    multi_perform(m, &running);

    abort_on_test_timeout();

    if(!running)
      break; /* done */

    FD_ZERO(&fdread);
    FD_ZERO(&fdwrite);
    FD_ZERO(&fdexcep);

    multi_fdset(m, &fdread, &fdwrite, &fdexcep, &maxfd);

    /* At this point, maxfd is guaranteed to be greater or equal than -1. */

    select_test(maxfd + 1, &fdread, &fdwrite, &fdexcep, &timeout);

    abort_on_test_timeout();
  }

test_cleanup:

  /* proper cleanup sequence - type PA */

  curl_multi_remove_handle(m, curl);
  curl_multi_cleanup(m);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res;
}
