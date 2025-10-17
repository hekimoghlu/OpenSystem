/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 1, 2025.
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

#define EXCESSIVE 10*1000*1000
int test(char *URL)
{
  CURLcode res = CURLE_OK;
  CURL *curl = NULL;
  char *longurl = malloc(EXCESSIVE);
  CURLU *u;
  (void)URL;

  if(!longurl)
    return 1;

  memset(longurl, 'a', EXCESSIVE);
  longurl[EXCESSIVE-1] = 0;

  global_init(CURL_GLOBAL_ALL);
  easy_init(curl);

  res = curl_easy_setopt(curl, CURLOPT_URL, longurl);
  printf("CURLOPT_URL %d bytes URL == %d\n",
         EXCESSIVE, (int)res);

  res = curl_easy_setopt(curl, CURLOPT_POSTFIELDS, longurl);
  printf("CURLOPT_POSTFIELDS %d bytes data == %d\n",
         EXCESSIVE, (int)res);

  u = curl_url();
  if(u) {
    CURLUcode uc = curl_url_set(u, CURLUPART_URL, longurl, 0);
    printf("CURLUPART_URL %d bytes URL == %d (%s)\n",
           EXCESSIVE, (int)uc, curl_url_strerror(uc));
    uc = curl_url_set(u, CURLUPART_SCHEME, longurl, CURLU_NON_SUPPORT_SCHEME);
    printf("CURLUPART_SCHEME %d bytes scheme == %d (%s)\n",
           EXCESSIVE, (int)uc, curl_url_strerror(uc));
    uc = curl_url_set(u, CURLUPART_USER, longurl, 0);
    printf("CURLUPART_USER %d bytes user == %d (%s)\n",
           EXCESSIVE, (int)uc, curl_url_strerror(uc));
    curl_url_cleanup(u);
  }

test_cleanup:
  free(longurl);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return res; /* return the final return code */
}
