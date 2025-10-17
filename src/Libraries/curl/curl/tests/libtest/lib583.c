/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 28, 2023.
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
 * This test case is based on the sample code provided by Saqib Ali
 * https://curl.se/mail/lib-2011-03/0066.html
 */

#include "test.h"

#include <sys/stat.h>

#include "memdebug.h"

int test(char *URL)
{
  int stillRunning;
  CURLM *multiHandle = NULL;
  CURL *curl = NULL;
  CURLcode res = CURLE_OK;
  CURLMcode mres;

  assert(test_argc >= 4);

  global_init(CURL_GLOBAL_ALL);

  multi_init(multiHandle);

  easy_init(curl);

  easy_setopt(curl, CURLOPT_USERPWD, libtest_arg2);
  easy_setopt(curl, CURLOPT_SSH_PUBLIC_KEYFILE,  test_argv[3]);
  easy_setopt(curl, CURLOPT_SSH_PRIVATE_KEYFILE, test_argv[4]);

  easy_setopt(curl, CURLOPT_UPLOAD, 1L);
  easy_setopt(curl, CURLOPT_VERBOSE, 1L);

  easy_setopt(curl, CURLOPT_URL, URL);
  easy_setopt(curl, CURLOPT_INFILESIZE, (long)5);

  multi_add_handle(multiHandle, curl);

  /* this tests if removing an easy handle immediately after multi
     perform has been called succeeds or not. */

  fprintf(stderr, "curl_multi_perform()...\n");

  multi_perform(multiHandle, &stillRunning);

  fprintf(stderr, "curl_multi_perform() succeeded\n");

  fprintf(stderr, "curl_multi_remove_handle()...\n");
  mres = curl_multi_remove_handle(multiHandle, curl);
  if(mres) {
    fprintf(stderr, "curl_multi_remove_handle() failed, "
            "with code %d\n", (int)mres);
    res = TEST_ERR_MULTI;
  }
  else
    fprintf(stderr, "curl_multi_remove_handle() succeeded\n");

test_cleanup:

  /* undocumented cleanup sequence - type UB */

  curl_easy_cleanup(curl);
  curl_multi_cleanup(multiHandle);
  curl_global_cleanup();

  return (int)res;
}
