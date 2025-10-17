/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 7, 2022.
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
/*
 * Test CURLOPT_MAXLIFETIME_CONN:
 * Send four requests, sleeping between the second and third and setting
 * MAXLIFETIME_CONN between the third and fourth. The first three requests
 * should use the same connection, and the fourth request should close the
 * first connection and open a second.
 */

#include "test.h"
#include "testutil.h"
#include "testtrace.h"
#include "warnless.h"
#include "memdebug.h"

int test(char *URL)
{
  CURL *easy = NULL;
  int res = 0;

  global_init(CURL_GLOBAL_ALL);

  res_easy_init(easy);

  easy_setopt(easy, CURLOPT_URL, URL);

  libtest_debug_config.nohex = 1;
  libtest_debug_config.tracetime = 0;
  easy_setopt(easy, CURLOPT_DEBUGDATA, &libtest_debug_config);
  easy_setopt(easy, CURLOPT_DEBUGFUNCTION, libtest_debug_cb);
  easy_setopt(easy, CURLOPT_VERBOSE, 1L);

  res = curl_easy_perform(easy);
  if(res)
    goto test_cleanup;

  res = curl_easy_perform(easy);
  if(res)
    goto test_cleanup;

  /* CURLOPT_MAXLIFETIME_CONN is inclusive - the connection needs to be 2
   * seconds old */
  sleep(2);

  res = curl_easy_perform(easy);
  if(res)
    goto test_cleanup;

  easy_setopt(easy, CURLOPT_MAXLIFETIME_CONN, 1L);

  res = curl_easy_perform(easy);
  if(res)
    goto test_cleanup;

test_cleanup:

  curl_easy_cleanup(easy);
  curl_global_cleanup();

  return res;
}
