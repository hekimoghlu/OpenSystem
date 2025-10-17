/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#include "memdebug.h"

int test(char *URL)
{
  CURLM *multi;
  CURL *easy;
  int running_handles;

  curl_global_init(CURL_GLOBAL_DEFAULT);

  multi = curl_multi_init();
  if(multi) {
    easy = curl_easy_init();
    if(easy) {
      CURLcode c;
      CURLMcode m;

      /* Crash only happens when using HTTPS */
      c = curl_easy_setopt(easy, CURLOPT_URL, URL);
      if(!c)
        /* Any old HTTP tunneling proxy will do here */
        c = curl_easy_setopt(easy, CURLOPT_PROXY, libtest_arg2);

      if(!c) {

        /* We're going to drive the transfer using multi interface here,
           because we want to stop during the middle. */
        m = curl_multi_add_handle(multi, easy);

        if(!m)
          /* Run the multi handle once, just enough to start establishing an
             HTTPS connection. */
          m = curl_multi_perform(multi, &running_handles);

        if(m)
          fprintf(stderr, "curl_multi_perform failed\n");
      }
      /* Close the easy handle *before* the multi handle. Doing it the other
         way around avoids the issue. */
      curl_easy_cleanup(easy);
    }
    curl_multi_cleanup(multi); /* double-free happens here */
  }
  curl_global_cleanup();
  return CURLE_OK;
}
