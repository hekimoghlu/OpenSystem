/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
  CURLM *curlm = NULL;
  CURL *curl1 = NULL;
  CURL *curl2 = NULL;
  int running_handles = 0;
  int res = 0;

  global_init(CURL_GLOBAL_ALL);

  multi_init(curlm);
  multi_setopt(curlm, CURLMOPT_MAX_HOST_CONNECTIONS, 1);

  easy_init(curl1);
  easy_setopt(curl1, CURLOPT_URL, URL);
  multi_add_handle(curlm, curl1);

  easy_init(curl2);
  easy_setopt(curl2, CURLOPT_URL, URL);
  multi_add_handle(curlm, curl2);

  multi_perform(curlm, &running_handles);

  multi_remove_handle(curlm, curl2);

  /* If curl2 is still in the connect-pending list, this will crash */
  multi_remove_handle(curlm, curl1);

test_cleanup:
  curl_easy_cleanup(curl1);
  curl_easy_cleanup(curl2);
  curl_multi_cleanup(curlm);
  curl_global_cleanup();
  return res;
}
