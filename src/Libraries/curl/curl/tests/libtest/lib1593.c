/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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
/* Test suppressing the If-Modified-Since header */

#include "test.h"

#include "memdebug.h"

int test(char *URL)
{
  struct curl_slist *header = NULL;
  long unmet;
  CURL *curl = NULL;
  int res = 0;

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  easy_setopt(curl, CURLOPT_URL, URL);
  easy_setopt(curl, CURLOPT_TIMECONDITION, (long)CURL_TIMECOND_IFMODSINCE);
  /* Some TIMEVALUE; it doesn't matter. */
  easy_setopt(curl, CURLOPT_TIMEVALUE, 1566210680L);

  header = curl_slist_append(NULL, "If-Modified-Since:");
  if(!header) {
    res = TEST_ERR_MAJOR_BAD;
    goto test_cleanup;
  }

  easy_setopt(curl, CURLOPT_HTTPHEADER, header);

  res = curl_easy_perform(curl);
  if(res)
    goto test_cleanup;

  /* Confirm that the condition checking still worked, even though we
   * suppressed the actual header.
   * The server returns 304, which means the condition is "unmet".
   */

  res = curl_easy_getinfo(curl, CURLINFO_CONDITION_UNMET, &unmet);
  if(res)
    goto test_cleanup;

  if(unmet != 1L) {
    res = TEST_ERR_FAILURE;
    goto test_cleanup;
  }

test_cleanup:

  /* always cleanup */
  curl_easy_cleanup(curl);
  curl_slist_free_all(header);
  curl_global_cleanup();

  return res;
}
