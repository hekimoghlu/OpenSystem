/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
  CURL *curl;
  int i;

  global_init(CURL_GLOBAL_ALL);
  easy_init(curl);
  easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
  easy_setopt(curl, CURLOPT_XOAUTH2_BEARER,
                   "c4e448d652a961fda0ab64f882c8c161d5985f805d45d80c9ddca1");
  easy_setopt(curl, CURLOPT_SASL_AUTHZID,
                   "c4e448d652a961fda0ab64f882c8c161d5985f805d45d80c9ddca2");
  easy_setopt(curl, CURLOPT_URL, URL);

  for(i = 0; i < 2; i++) {
    /* the second request needs to do connection reuse */
    res = curl_easy_perform(curl);
    if(res)
      goto test_cleanup;
  }

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();
  return (int)res;
}
