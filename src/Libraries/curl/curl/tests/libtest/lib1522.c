/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

/* test case and code based on https://github.com/curl/curl/issues/2847 */

#include "testtrace.h"
#include "testutil.h"
#include "warnless.h"
#include "memdebug.h"

static char g_Data[40 * 1024]; /* POST 40KB */

static int sockopt_callback(void *clientp, curl_socket_t curlfd,
                            curlsocktype purpose)
{
#if defined(SOL_SOCKET) && defined(SO_SNDBUF)
  int sndbufsize = 4 * 1024; /* 4KB send buffer */
  (void) clientp;
  (void) purpose;
  setsockopt(curlfd, SOL_SOCKET, SO_SNDBUF,
             (char *)&sndbufsize, sizeof(sndbufsize));
#else
  (void)clientp;
  (void)curlfd;
  (void)purpose;
#endif
  return CURL_SOCKOPT_OK;
}

int test(char *URL)
{
  CURLcode code = TEST_ERR_MAJOR_BAD;
  CURLcode res;
  struct curl_slist *pHeaderList = NULL;
  CURL *curl = curl_easy_init();
  memset(g_Data, 'A', sizeof(g_Data)); /* send As! */

  curl_easy_setopt(curl, CURLOPT_SOCKOPTFUNCTION, sockopt_callback);
  curl_easy_setopt(curl, CURLOPT_URL, URL);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, g_Data);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, (long)sizeof(g_Data));

  libtest_debug_config.nohex = 1;
  libtest_debug_config.tracetime = 1;
  test_setopt(curl, CURLOPT_DEBUGDATA, &libtest_debug_config);
  test_setopt(curl, CURLOPT_DEBUGFUNCTION, libtest_debug_cb);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  /* Remove "Expect: 100-continue" */
  pHeaderList = curl_slist_append(pHeaderList, "Expect:");

  curl_easy_setopt(curl, CURLOPT_HTTPHEADER, pHeaderList);

  code = curl_easy_perform(curl);

  if(code == CURLE_OK) {
    curl_off_t uploadSize;
    curl_easy_getinfo(curl, CURLINFO_SIZE_UPLOAD_T, &uploadSize);

    printf("uploadSize = %ld\n", (long)uploadSize);

    if((size_t) uploadSize == sizeof(g_Data)) {
      printf("!!!!!!!!!! PASS\n");
    }
    else {
      printf("sent %d, libcurl says %d\n",
             (int)sizeof(g_Data), (int)uploadSize);
    }
  }
  else {
    printf("curl_easy_perform() failed. e = %d\n", code);
  }
test_cleanup:
  curl_slist_free_all(pHeaderList);
  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)code;
}
