/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 16, 2023.
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
  char *url_after;
  CURL *curl;
  CURLcode res = CURLE_OK;
  char error_buffer[CURL_ERROR_SIZE] = "";

  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, URL);
  curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  res = curl_easy_perform(curl);
  if(!res)
    fprintf(stderr, "failure expected, "
            "curl_easy_perform returned %ld: <%s>, <%s>\n",
            (long) res, curl_easy_strerror(res), error_buffer);

  /* print the used url */
  if(!curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &url_after))
    printf("Effective URL: %s\n", url_after);

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)res;
}
