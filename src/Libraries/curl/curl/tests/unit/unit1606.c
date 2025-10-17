/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 5, 2023.
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

#include "speedcheck.h"
#include "urldata.h"

static CURL *easy;

static CURLcode unit_setup(void)
{
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);
  easy = curl_easy_init();
  if(!easy) {
    curl_global_cleanup();
    return CURLE_OUT_OF_MEMORY;
  }
  return res;
}

static void unit_stop(void)
{
  curl_easy_cleanup(easy);
  curl_global_cleanup();
}

static int runawhile(long time_limit,
                     long speed_limit,
                     curl_off_t speed,
                     int dec)
{
  int counter = 1;
  struct curltime now = {1, 0};
  CURLcode result;
  int finaltime;

  curl_easy_setopt(easy, CURLOPT_LOW_SPEED_LIMIT, speed_limit);
  curl_easy_setopt(easy, CURLOPT_LOW_SPEED_TIME, time_limit);
  Curl_speedinit(easy);

  do {
    /* fake the current transfer speed */
    easy->progress.current_speed = speed;
    result = Curl_speedcheck(easy, now);
    if(result)
      break;
    /* step the time */
    now.tv_sec = ++counter;
    speed -= dec;
  } while(counter < 100);

  finaltime = (int)(now.tv_sec - 1);

  return finaltime;
}

UNITTEST_START
  fail_unless(runawhile(41, 41, 40, 0) == 41,
              "wrong low speed timeout");
  fail_unless(runawhile(21, 21, 20, 0) == 21,
              "wrong low speed timeout");
  fail_unless(runawhile(60, 60, 40, 0) == 60,
              "wrong log speed timeout");
  fail_unless(runawhile(50, 50, 40, 0) == 50,
              "wrong log speed timeout");
  fail_unless(runawhile(40, 40, 40, 0) == 99,
              "should not time out");
  fail_unless(runawhile(10, 50, 100, 2) == 36,
              "bad timeout");
UNITTEST_STOP
