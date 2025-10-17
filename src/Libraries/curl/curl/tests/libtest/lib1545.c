/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 13, 2022.
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
#ifndef CURL_DISABLE_DEPRECATION
#define CURL_DISABLE_DEPRECATION  /* Using and testing the form api */
#endif
#include "test.h"

int test(char *URL)
{
  CURL *eh = NULL;
  int res = 0;
  struct curl_httppost *lastptr = NULL;
  struct curl_httppost *m_formpost = NULL;

  global_init(CURL_GLOBAL_ALL);

  easy_init(eh);

  easy_setopt(eh, CURLOPT_URL, URL);
  curl_formadd(&m_formpost, &lastptr, CURLFORM_COPYNAME, "file",
               CURLFORM_FILE, "missing-file", CURLFORM_END);
  curl_easy_setopt(eh, CURLOPT_HTTPPOST, m_formpost);

  (void)curl_easy_perform(eh);
  (void)curl_easy_perform(eh);

test_cleanup:

  curl_formfree(m_formpost);

  curl_easy_cleanup(eh);
  curl_global_cleanup();

  return res;
}
