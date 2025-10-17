/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 21, 2025.
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

#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

int test(char *URL)
{
  CURLcode res = CURLE_OK;
  char *url_after = NULL;
  CURLU *curlu = curl_url();
  char error_buffer[CURL_ERROR_SIZE] = "";
  CURL *curl;

  easy_init(curl);

  curl_url_set(curlu, CURLUPART_URL, URL, CURLU_DEFAULT_SCHEME);
  easy_setopt(curl, CURLOPT_CURLU, curlu);
  easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
  easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  /* msys2 times out instead of CURLE_COULDNT_CONNECT, so make it faster */
  easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
  /* set a port number that makes this request fail */
  easy_setopt(curl, CURLOPT_PORT, 1L);
  res = curl_easy_perform(curl);
  if(res != CURLE_COULDNT_CONNECT && res != CURLE_OPERATION_TIMEDOUT) {
    fprintf(stderr, "failure expected, "
            "curl_easy_perform returned %d: <%s>, <%s>\n",
            (int) res, curl_easy_strerror(res), error_buffer);
    if(res == CURLE_OK)
      res = TEST_ERR_MAJOR_BAD;  /* force an error return */
    goto test_cleanup;
  }
  res = CURLE_OK;  /* reset for next use */

  /* print the used url */
  curl_url_get(curlu, CURLUPART_URL, &url_after, 0);
  fprintf(stderr, "curlu now: <%s>\n", url_after);
  curl_free(url_after);
  url_after = NULL;

  /* now reset CURLOP_PORT to go back to originally set port number */
  easy_setopt(curl, CURLOPT_PORT, 0L);

  res = curl_easy_perform(curl);
  if(res)
    fprintf(stderr, "success expected, "
            "curl_easy_perform returned %ld: <%s>, <%s>\n",
            (long) res, curl_easy_strerror(res), error_buffer);

  /* print url */
  curl_url_get(curlu, CURLUPART_URL, &url_after, 0);
  fprintf(stderr, "curlu now: <%s>\n", url_after);

test_cleanup:
  curl_free(url_after);
  curl_easy_cleanup(curl);
  curl_url_cleanup(curlu);
  curl_global_cleanup();

  return (int)res;
}
