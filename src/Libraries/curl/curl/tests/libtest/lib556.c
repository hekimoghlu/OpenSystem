/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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

#include "warnless.h"
#include "memdebug.h"

/* For Windows, mainly (may be moved in a config file?) */
#ifndef STDIN_FILENO
  #define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
  #define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
  #define STDERR_FILENO 2
#endif

int test(char *URL)
{
  CURLcode res;
  CURL *curl;

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
  test_setopt(curl, CURLOPT_CONNECT_ONLY, 1L);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  res = curl_easy_perform(curl);

  if(!res) {
    /* we are connected, now get an HTTP document the raw way */
    const char *request =
      "GET /556 HTTP/1.1\r\n"
      "Host: ninja\r\n\r\n";
    size_t iolen = 0;

    res = curl_easy_send(curl, request, strlen(request), &iolen);

    if(!res) {
      /* we assume that sending always work */

      do {
        char buf[1024];
        /* busy-read like crazy */
        res = curl_easy_recv(curl, buf, sizeof(buf), &iolen);

        if(iolen) {
          /* send received stuff to stdout */
          if(!write(STDOUT_FILENO, buf, iolen))
            break;
        }

      } while((res == CURLE_OK && iolen) || (res == CURLE_AGAIN));
    }

    if(iolen)
      res = (CURLcode)TEST_ERR_FAILURE;
  }

test_cleanup:

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)res;
}
