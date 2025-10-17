/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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

struct transfer_status {
  CURL *easy;
  int hd_count;
  int bd_count;
  CURLcode result;
};

#define KN(a)   a, #a

static int geterr(const char *name, CURLcode val, int lineno)
{
  printf("CURLINFO_%s returned %d, \"%s\" on line %d\n",
         name, val, curl_easy_strerror(val), lineno);
  return (int)val;
}

static void report_time(const char *key, const char *where, curl_off_t time,
                        bool ok)
{
  if(ok)
    printf("%s on %s is OK\n", key, where);
  else
    printf("%s on %s is WRONG: %" CURL_FORMAT_CURL_OFF_T "\n",
           key, where, time);
}

static void check_time(CURL *easy, int key, const char *name,
                       const char *where)
{
  curl_off_t tval;
  CURLcode res = curl_easy_getinfo(easy, (CURLINFO)key, &tval);
  if(res) {
    geterr(name, res, __LINE__);
  }
  else
    report_time(name, where, tval, tval > 0);
}

static void check_time0(CURL *easy, int key, const char *name,
                        const char *where)
{
  curl_off_t tval;
  CURLcode res = curl_easy_getinfo(easy, (CURLINFO)key, &tval);
  if(res) {
    geterr(name, res, __LINE__);
  }
  else
    report_time(name, where, tval, !tval);
}

static size_t header_callback(void *ptr, size_t size, size_t nmemb,
                              void *userp)
{
  struct transfer_status *st = (struct transfer_status *)userp;
  size_t len = size * nmemb;

  (void)ptr;
  if(!st->hd_count++) {
    /* first header, check some CURLINFO value to be reported. See #13125 */
    check_time(st->easy, KN(CURLINFO_CONNECT_TIME_T), "1st header");
    check_time(st->easy, KN(CURLINFO_PRETRANSFER_TIME_T), "1st header");
    check_time(st->easy, KN(CURLINFO_STARTTRANSFER_TIME_T), "1st header");
    /* continuously updated */
    check_time(st->easy, KN(CURLINFO_TOTAL_TIME_T), "1st header");
    /* no SSL, must be 0 */
    check_time0(st->easy, KN(CURLINFO_APPCONNECT_TIME_T), "1st header");
    /* download not really started */
    check_time0(st->easy, KN(CURLINFO_SPEED_DOWNLOAD_T), "1st header");
  }
  (void)fwrite(ptr, size, nmemb, stdout);
  return len;
}

static size_t write_callback(void *ptr, size_t size, size_t nmemb, void *userp)
{
  struct transfer_status *st = (struct transfer_status *)userp;

  (void)ptr;
  (void)st;
  fwrite(ptr, size, nmemb, stdout);
  return size * nmemb;
}

int test(char *URL)
{
  CURL *curls = NULL;
  int res = 0;
  struct transfer_status st;

  start_test_timing();

  memset(&st, 0, sizeof(st));

  global_init(CURL_GLOBAL_ALL);

  easy_init(curls);
  st.easy = curls; /* to allow callbacks access */

  easy_setopt(curls, CURLOPT_URL, URL);
  easy_setopt(curls, CURLOPT_WRITEFUNCTION, write_callback);
  easy_setopt(curls, CURLOPT_WRITEDATA, &st);
  easy_setopt(curls, CURLOPT_HEADERFUNCTION, header_callback);
  easy_setopt(curls, CURLOPT_HEADERDATA, &st);

  easy_setopt(curls, CURLOPT_NOPROGRESS, 0L);

  res = curl_easy_perform(curls);

  check_time(curls, KN(CURLINFO_CONNECT_TIME_T), "done");
  check_time(curls, KN(CURLINFO_PRETRANSFER_TIME_T), "done");
  check_time(curls, KN(CURLINFO_STARTTRANSFER_TIME_T), "done");
  /* no SSL, must be 0 */
  check_time0(curls, KN(CURLINFO_APPCONNECT_TIME_T), "done");
  check_time(curls, KN(CURLINFO_SPEED_DOWNLOAD_T), "done");
  check_time(curls, KN(CURLINFO_TOTAL_TIME_T), "done");

test_cleanup:

  curl_easy_cleanup(curls);
  curl_global_cleanup();

  return (int)res; /* return the final return code */
}
