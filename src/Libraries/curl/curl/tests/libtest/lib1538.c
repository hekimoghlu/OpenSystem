/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
  int res = 0;
  CURLcode easyret;
  CURLMcode multiret;
  CURLSHcode shareret;
  CURLUcode urlret;
  (void)URL;

  curl_easy_strerror((CURLcode)INT_MAX);
  curl_multi_strerror((CURLMcode)INT_MAX);
  curl_share_strerror((CURLSHcode)INT_MAX);
  curl_url_strerror((CURLUcode)INT_MAX);
  curl_easy_strerror((CURLcode)-INT_MAX);
  curl_multi_strerror((CURLMcode)-INT_MAX);
  curl_share_strerror((CURLSHcode)-INT_MAX);
  curl_url_strerror((CURLUcode)-INT_MAX);
  for(easyret = CURLE_OK; easyret <= CURL_LAST; easyret++) {
    printf("e%d: %s\n", (int)easyret, curl_easy_strerror(easyret));
  }
  for(multiret = CURLM_CALL_MULTI_PERFORM; multiret <= CURLM_LAST;
      multiret++) {
    printf("m%d: %s\n", (int)multiret, curl_multi_strerror(multiret));
  }
  for(shareret = CURLSHE_OK; shareret <= CURLSHE_LAST; shareret++) {
    printf("s%d: %s\n", (int)shareret, curl_share_strerror(shareret));
  }
  for(urlret = CURLUE_OK; urlret <= CURLUE_LAST; urlret++) {
    printf("u%d: %s\n", (int)urlret, curl_url_strerror(urlret));
  }

  return (int)res;
}
