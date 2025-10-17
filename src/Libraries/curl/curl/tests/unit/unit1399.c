/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 17, 2023.
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

#include "urldata.h"
#include "progress.h"

static int usec_magnitude = 1000000;

static bool unit_setup(void)
{
  return CURLE_OK;
}

static void unit_stop(void)
{

}

/*
 * Invoke Curl_pgrsTime for TIMER_STARTSINGLE to trigger the behavior that
 * manages is_t_startransfer_set, but fake the t_startsingle time for purposes
 * of the test.
 */
static void fake_t_startsingle_time(struct Curl_easy *data,
                                    struct curltime fake_now,
                                    int seconds_offset)
{
  Curl_pgrsTime(data, TIMER_STARTSINGLE);
  data->progress.t_startsingle.tv_sec = fake_now.tv_sec + seconds_offset;
  data->progress.t_startsingle.tv_usec = fake_now.tv_usec;
}

static bool usec_matches_seconds(timediff_t time_usec, int expected_seconds)
{
  int time_sec = (int)(time_usec / usec_magnitude);
  bool same = (time_sec == expected_seconds);
  fprintf(stderr, "is %d us same as %d seconds? %s\n",
          (int)time_usec, expected_seconds,
          same?"Yes":"No");
  return same;
}

static void expect_timer_seconds(struct Curl_easy *data, int seconds)
{
  char msg[64];
  msnprintf(msg, sizeof(msg), "about %d seconds should have passed", seconds);
  fail_unless(usec_matches_seconds(data->progress.t_nslookup, seconds), msg);
  fail_unless(usec_matches_seconds(data->progress.t_connect, seconds), msg);
  fail_unless(usec_matches_seconds(data->progress.t_appconnect, seconds), msg);
  fail_unless(usec_matches_seconds(data->progress.t_pretransfer, seconds),
              msg);
  fail_unless(usec_matches_seconds(data->progress.t_starttransfer, seconds),
              msg);
}

/* Scenario: simulate a redirect. When a redirect occurs, t_nslookup,
 * t_connect, t_appconnect, t_pretransfer, and t_starttransfer are additive.
 * E.g., if t_starttransfer took 2 seconds initially and took another 1
 * second for the redirect request, then the resulting t_starttransfer should
 * be 3 seconds. */
UNITTEST_START
  struct Curl_easy data;
  struct curltime now = Curl_now();

  data.progress.t_nslookup = 0;
  data.progress.t_connect = 0;
  data.progress.t_appconnect = 0;
  data.progress.t_pretransfer = 0;
  data.progress.t_starttransfer = 0;
  data.progress.t_redirect = 0;
  data.progress.start.tv_sec = now.tv_sec - 2;
  data.progress.start.tv_usec = now.tv_usec;
  fake_t_startsingle_time(&data, now, -2);

  Curl_pgrsTime(&data, TIMER_NAMELOOKUP);
  Curl_pgrsTime(&data, TIMER_CONNECT);
  Curl_pgrsTime(&data, TIMER_APPCONNECT);
  Curl_pgrsTime(&data, TIMER_PRETRANSFER);
  Curl_pgrsTime(&data, TIMER_STARTTRANSFER);

  expect_timer_seconds(&data, 2);

  /* now simulate the redirect */
  data.progress.t_redirect = data.progress.t_starttransfer + 1;
  fake_t_startsingle_time(&data, now, -1);

  Curl_pgrsTime(&data, TIMER_NAMELOOKUP);
  Curl_pgrsTime(&data, TIMER_CONNECT);
  Curl_pgrsTime(&data, TIMER_APPCONNECT);
  Curl_pgrsTime(&data, TIMER_PRETRANSFER);
  /* ensure t_starttransfer is only set on the first invocation by attempting
   * to set it twice */
  Curl_pgrsTime(&data, TIMER_STARTTRANSFER);
  Curl_pgrsTime(&data, TIMER_STARTTRANSFER);

  expect_timer_seconds(&data, 3);
UNITTEST_STOP
