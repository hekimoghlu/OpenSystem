/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 1, 2023.
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

#include "testtrace.h"
#include "memdebug.h"

#ifdef LIB585

static int counter;

static curl_socket_t tst_opensocket(void *clientp,
                                    curlsocktype purpose,
                                    struct curl_sockaddr *addr)
{
  (void)clientp;
  (void)purpose;
  printf("[OPEN] counter: %d\n", ++counter);
  return socket(addr->family, addr->socktype, addr->protocol);
}

static int tst_closesocket(void *clientp, curl_socket_t sock)
{
  (void)clientp;
  printf("[CLOSE] counter: %d\n", counter--);
  return sclose(sock);
}

static void setupcallbacks(CURL *curl)
{
  curl_easy_setopt(curl, CURLOPT_OPENSOCKETFUNCTION, tst_opensocket);
  curl_easy_setopt(curl, CURLOPT_CLOSESOCKETFUNCTION, tst_closesocket);
  counter = 0;
}

#else
#define setupcallbacks(x) Curl_nop_stmt
#endif


int test(char *URL)
{
  CURLcode res;
  CURL *curl;
  char *ipstr = NULL;

  if(curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
    fprintf(stderr, "curl_global_init() failed\n");
    return TEST_ERR_MAJOR_BAD;
  }

  curl = curl_easy_init();
  if(!curl) {
    fprintf(stderr, "curl_easy_init() failed\n");
    curl_global_cleanup();
    return TEST_ERR_MAJOR_BAD;
  }

  test_setopt(curl, CURLOPT_URL, URL);
  test_setopt(curl, CURLOPT_HEADER, 1L);

  libtest_debug_config.nohex = 1;
  libtest_debug_config.tracetime = 1;
  test_setopt(curl, CURLOPT_DEBUGDATA, &libtest_debug_config);
  test_setopt(curl, CURLOPT_DEBUGFUNCTION, libtest_debug_cb);
  test_setopt(curl, CURLOPT_VERBOSE, 1L);

  if(libtest_arg3 && !strcmp(libtest_arg3, "activeftp"))
    test_setopt(curl, CURLOPT_FTPPORT, "-");

  setupcallbacks(curl);

  res = curl_easy_perform(curl);

  if(!res) {
    res = curl_easy_getinfo(curl, CURLINFO_PRIMARY_IP, &ipstr);
    if(libtest_arg2) {
      FILE *moo = fopen(libtest_arg2, "wb");
      if(moo) {
        curl_off_t time_namelookup;
        curl_off_t time_connect;
        curl_off_t time_pretransfer;
        curl_off_t time_starttransfer;
        curl_off_t time_total;
        fprintf(moo, "IP %s\n", ipstr);
        curl_easy_getinfo(curl, CURLINFO_NAMELOOKUP_TIME_T, &time_namelookup);
        curl_easy_getinfo(curl, CURLINFO_CONNECT_TIME_T, &time_connect);
        curl_easy_getinfo(curl, CURLINFO_PRETRANSFER_TIME_T,
                          &time_pretransfer);
        curl_easy_getinfo(curl, CURLINFO_STARTTRANSFER_TIME_T,
                          &time_starttransfer);
        curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME_T, &time_total);

        /* since the timing will always vary we only compare relative
           differences between these 5 times */
        if(time_namelookup > time_connect) {
          fprintf(moo, "namelookup vs connect: %" CURL_FORMAT_CURL_OFF_T
                  ".%06ld %" CURL_FORMAT_CURL_OFF_T ".%06ld\n",
                  (time_namelookup / 1000000),
                  (long)(time_namelookup % 1000000),
                  (time_connect / 1000000), (long)(time_connect % 1000000));
        }
        if(time_connect > time_pretransfer) {
          fprintf(moo, "connect vs pretransfer: %" CURL_FORMAT_CURL_OFF_T
                  ".%06ld %" CURL_FORMAT_CURL_OFF_T ".%06ld\n",
                  (time_connect / 1000000), (long)(time_connect % 1000000),
                  (time_pretransfer / 1000000),
                  (long)(time_pretransfer % 1000000));
        }
        if(time_pretransfer > time_starttransfer) {
          fprintf(moo, "pretransfer vs starttransfer: %" CURL_FORMAT_CURL_OFF_T
                  ".%06ld %" CURL_FORMAT_CURL_OFF_T ".%06ld\n",
                  (time_pretransfer / 1000000),
                  (long)(time_pretransfer % 1000000),
                  (time_starttransfer / 1000000),
                  (long)(time_starttransfer % 1000000));
        }
        if(time_starttransfer > time_total) {
          fprintf(moo, "starttransfer vs total: %" CURL_FORMAT_CURL_OFF_T
                  ".%06ld %" CURL_FORMAT_CURL_OFF_T ".%06ld\n",
                  (time_starttransfer / 1000000),
                  (long)(time_starttransfer % 1000000),
                  (time_total / 1000000), (long)(time_total % 1000000));
        }

        fclose(moo);
      }
    }
  }

test_cleanup:

  curl_easy_cleanup(curl);
  curl_global_cleanup();

  return (int)res;
}
