/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#include "curlcheck.h"

static CURL *hnd;

static CURLcode unit_setup(void)
{
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);
  return res;
}

static void unit_stop(void)
{
  if(hnd)
    curl_easy_cleanup(hnd);
  curl_global_cleanup();
}

struct test {
  const char *in;
  int inlen;
  const char *out;
  int outlen;
};

UNITTEST_START
{
  /* unescape, this => that */
  const struct test list1[]={
    {"%61", 3, "a", 1},
    {"%61a", 4, "aa", 2},
    {"%61b", 4, "ab", 2},
    {"%6 1", 4, "%6 1", 4},
    {"%61", 1, "%", 1},
    {"%61", 2, "%6", 2},
    {"%6%a", 4, "%6%a", 4},
    {"%6a", 0, "j", 1},
    {"%FF", 0, "\xff", 1},
    {"%FF%00%ff", 9, "\xff\x00\xff", 3},
    {"%-2", 0, "%-2", 3},
    {"%FG", 0, "%FG", 3},
    {NULL, 0, NULL, 0} /* end of list marker */
  };
  /* escape, this => that */
  const struct test list2[]={
    {"a", 1, "a", 1},
    {"/", 1, "%2F", 3},
    {"a=b", 3, "a%3Db", 5},
    {"a=b", 0, "a%3Db", 5},
    {"a=b", 1, "a", 1},
    {"a=b", 2, "a%3D", 4},
    {"1/./0", 5, "1%2F.%2F0", 9},
    {"-._~!#%&", 0, "-._~%21%23%25%26", 16},
    {"a", 2, "a%00", 4},
    {"a\xff\x01g", 4, "a%FF%01g", 8},
    {NULL, 0, NULL, 0} /* end of list marker */
  };
  int i;

  hnd = curl_easy_init();
  abort_unless(hnd != NULL, "returned NULL!");
  for(i = 0; list1[i].in; i++) {
    int outlen;
    char *out = curl_easy_unescape(hnd,
                                   list1[i].in, list1[i].inlen,
                                   &outlen);

    abort_unless(out != NULL, "returned NULL!");
    fail_unless(outlen == list1[i].outlen, "wrong output length returned");
    fail_unless(!memcmp(out, list1[i].out, list1[i].outlen),
                "bad output data returned");

    printf("curl_easy_unescape test %d DONE\n", i);

    curl_free(out);
  }

  for(i = 0; list2[i].in; i++) {
    int outlen;
    char *out = curl_easy_escape(hnd, list2[i].in, list2[i].inlen);
    abort_unless(out != NULL, "returned NULL!");

    outlen = (int)strlen(out);
    fail_unless(outlen == list2[i].outlen, "wrong output length returned");
    fail_unless(!memcmp(out, list2[i].out, list2[i].outlen),
                "bad output data returned");

    printf("curl_easy_escape test %d DONE (%s)\n", i, out);

    curl_free(out);
  }
}
UNITTEST_STOP
