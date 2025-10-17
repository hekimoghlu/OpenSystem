/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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

#include "timeval.h"

static CURLcode unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

struct a {
  struct curltime first;
  struct curltime second;
  time_t result;
};

UNITTEST_START
{
  struct a tests[] = {
    { {36762, 8345 }, {36761, 995926 }, 13 },
    { {36761, 995926 }, {36762, 8345 }, -13 },
    { {36761, 995926 }, {0, 0}, 36761995 },
    { {0, 0}, {36761, 995926 }, -36761995 },
  };
  size_t i;

  for(i = 0; i < sizeof(tests)/sizeof(tests[0]); i++) {
    timediff_t result = Curl_timediff(tests[i].first, tests[i].second);
    if(result != tests[i].result) {
      printf("%ld.%06u to %ld.%06u got %d, but expected %ld\n",
             (long)tests[i].first.tv_sec,
             tests[i].first.tv_usec,
             (long)tests[i].second.tv_sec,
             tests[i].second.tv_usec,
             (int)result,
             (long)tests[i].result);
      fail("unexpected result!");
    }
  }
}
UNITTEST_STOP
