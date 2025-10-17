/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 20, 2023.
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
/* Testing CURLOPT_PROTOCOLS_STR */

#include "test.h"

#include "memdebug.h"

struct pair {
  const char *in;
  CURLcode *exp;
};

int test(char *URL)
{
  CURL *curl = NULL;
  int res = 0;
  CURLcode result = CURLE_OK;
  curl_version_info_data *curlinfo;
  const char *const *proto;
  int n;
  int i;
  static CURLcode ok = CURLE_OK;
  static CURLcode bad = CURLE_BAD_FUNCTION_ARGUMENT;
  static CURLcode unsup = CURLE_UNSUPPORTED_PROTOCOL;
  static CURLcode httpcode = CURLE_UNSUPPORTED_PROTOCOL;
  static CURLcode httpscode = CURLE_UNSUPPORTED_PROTOCOL;
  static char protolist[1024];

  static const struct pair prots[] = {
    {"goobar", &unsup},
    {"http ", &unsup},
    {" http", &unsup},
    {"http", &httpcode},
    {"http,", &httpcode},
    {"https,", &httpscode},
    {"https,http", &httpscode},
    {"http,http", &httpcode},
    {"HTTP,HTTP", &httpcode},
    {",HTTP,HTTP", &httpcode},
    {"http,http,ft", &unsup},
    {"", &bad},
    {",,", &bad},
    {protolist, &ok},
    {"all", &ok},
    {NULL, NULL},
  };
  (void)URL;

  global_init(CURL_GLOBAL_ALL);

  easy_init(curl);

  /* Get enabled protocols.*/
  curlinfo = curl_version_info(CURLVERSION_NOW);
  if(!curlinfo) {
    fputs("curl_version_info failed\n", stderr);
    res = (int) TEST_ERR_FAILURE;
    goto test_cleanup;
  }

  n = 0;
  for(proto = curlinfo->protocols; *proto; proto++) {
    if((size_t) n >= sizeof(protolist)) {
      puts("protolist buffer too small\n");
      res = (int) TEST_ERR_FAILURE;
      goto test_cleanup;
    }
    n += msnprintf(protolist + n, sizeof(protolist) - n, ",%s", *proto);
    if(curl_strequal(*proto, "http"))
      httpcode = CURLE_OK;
    if(curl_strequal(*proto, "https"))
      httpscode = CURLE_OK;
  }

  /* Run the tests. */
  for(i = 0; prots[i].in; i++) {
    result = curl_easy_setopt(curl, CURLOPT_PROTOCOLS_STR, prots[i].in);
    if(result != *prots[i].exp) {
      printf("unexpectedly '%s' returned %u\n",
             prots[i].in, result);
      break;
    }
  }
  printf("Tested %u strings\n", i);
  res = (int)result;

test_cleanup:
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)result;
}
