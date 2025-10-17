/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 15, 2024.
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

/*
 * Get a single URL without select().
 */

int test(char *URL)
{
  CURL *handle = NULL;
  CURLcode res = CURLE_OK;
  CURLU *urlp = NULL;

  global_init(CURL_GLOBAL_ALL);
  easy_init(handle);

  urlp = curl_url();

  if(!urlp) {
    fprintf(stderr, "problem init URL api.");
    goto test_cleanup;
  }

  /* this doesn't set the PATH part */
  if(curl_url_set(urlp, CURLUPART_HOST, "www.example.com", 0) ||
     curl_url_set(urlp, CURLUPART_SCHEME, "http", 0) ||
     curl_url_set(urlp, CURLUPART_PORT, "80", 0)) {
    fprintf(stderr, "problem setting CURLUPART");
    goto test_cleanup;
  }

  easy_setopt(handle, CURLOPT_CURLU, urlp);
  easy_setopt(handle, CURLOPT_VERBOSE, 1L);
  easy_setopt(handle, CURLOPT_PROXY, URL);

  res = curl_easy_perform(handle);

  if(res) {
    fprintf(stderr, "%s:%d curl_easy_perform() failed with code %d (%s)\n",
            __FILE__, __LINE__, res, curl_easy_strerror(res));
    goto test_cleanup;
  }

test_cleanup:

  curl_url_cleanup(urlp);
  curl_easy_cleanup(handle);
  curl_global_cleanup();

  return res;
}
