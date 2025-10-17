/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include "connect.h"
#include "memdebug.h" /* LAST include file */

static struct Curl_easy *data;

static CURLcode unit_setup(void)
{
  CURLcode res = CURLE_OK;

  global_init(CURL_GLOBAL_ALL);
  data = curl_easy_init();
  if(!data) {
    curl_global_cleanup();
    return CURLE_OUT_OF_MEMORY;
  }
  return res;
}

static void unit_stop(void)
{
  curl_easy_cleanup(data);
  curl_global_cleanup();
}

/* BASE is just a define to make us fool around with decently large number so
   that we aren't zero-based */
#define BASE 1000000

/* macro to set the pretended current time */
#define NOW(x,y) now.tv_sec = x; now.tv_usec = y
/* macro to set the millisecond based timeouts to use */
#define TIMEOUTS(x,y) data->set.timeout = x; data->set.connecttimeout = y

/*
 * To test:
 *
 * 00/10/01/11 timeouts set
 * 0/1         during connect
 * T           various values on the timeouts
 * N           various values of now
 */

struct timetest {
  int now_s;
  int now_us;
  int timeout_ms;
  int connecttimeout_ms;
  bool connecting;
  timediff_t result;
  const char *comment;
};

UNITTEST_START
{
  struct curltime now;
  unsigned int i;

  const struct timetest run[] = {
  /* both timeouts set, not connecting */
  {BASE + 4, 0,      10000, 8000, FALSE, 6000, "6 seconds should be left"},
  {BASE + 4, 990000, 10000, 8000, FALSE, 5010, "5010 ms should be left"},
  {BASE + 10, 0,     10000, 8000, FALSE, -1,   "timeout is -1, expired"},
  {BASE + 12, 0,     10000, 8000, FALSE, -2000, "-2000, overdue 2 seconds"},

  /* both timeouts set, connecting */
  {BASE + 4, 0,      10000, 8000, TRUE, 4000, "4 seconds should be left"},
  {BASE + 4, 990000, 10000, 8000, TRUE, 3010, "3010 ms should be left"},
  {BASE + 8, 0,      10000, 8000, TRUE, -1,   "timeout is -1, expired"},
  {BASE + 10, 0,     10000, 8000, TRUE, -2000, "-2000, overdue 2 seconds"},

  /* no connect timeout set, not connecting */
  {BASE + 4, 0,      10000, 0, FALSE, 6000, "6 seconds should be left"},
  {BASE + 4, 990000, 10000, 0, FALSE, 5010, "5010 ms should be left"},
  {BASE + 10, 0,     10000, 0, FALSE, -1,   "timeout is -1, expired"},
  {BASE + 12, 0,     10000, 0, FALSE, -2000, "-2000, overdue 2 seconds"},

  /* no connect timeout set, connecting */
  {BASE + 4, 0,      10000, 0, TRUE, 6000, "6 seconds should be left"},
  {BASE + 4, 990000, 10000, 0, TRUE, 5010, "5010 ms should be left"},
  {BASE + 10, 0,     10000, 0, TRUE, -1,   "timeout is -1, expired"},
  {BASE + 12, 0,     10000, 0, TRUE, -2000, "-2000, overdue 2 seconds"},

  /* only connect timeout set, not connecting */
  {BASE + 4, 0,      0, 10000, FALSE, 0, "no timeout active"},
  {BASE + 4, 990000, 0, 10000, FALSE, 0, "no timeout active"},
  {BASE + 10, 0,     0, 10000, FALSE, 0, "no timeout active"},
  {BASE + 12, 0,     0, 10000, FALSE, 0, "no timeout active"},

  /* only connect timeout set, connecting */
  {BASE + 4, 0,      0, 10000, TRUE, 6000, "6 seconds should be left"},
  {BASE + 4, 990000, 0, 10000, TRUE, 5010, "5010 ms should be left"},
  {BASE + 10, 0,     0, 10000, TRUE, -1,   "timeout is -1, expired"},
  {BASE + 12, 0,     0, 10000, TRUE, -2000, "-2000, overdue 2 seconds"},

  /* no timeout set, not connecting */
  {BASE + 4, 0,      0, 0, FALSE, 0, "no timeout active"},
  {BASE + 4, 990000, 0, 0, FALSE, 0, "no timeout active"},
  {BASE + 10, 0,     0, 0, FALSE, 0, "no timeout active"},
  {BASE + 12, 0,     0, 0, FALSE, 0, "no timeout active"},

  /* no timeout set, connecting */
  {BASE + 4, 0,      0, 0, TRUE, 296000, "no timeout active"},
  {BASE + 4, 990000, 0, 0, TRUE, 295010, "no timeout active"},
  {BASE + 10, 0,     0, 0, TRUE, 290000, "no timeout active"},
  {BASE + 12, 0,     0, 0, TRUE, 288000, "no timeout active"},

  /* both timeouts set, connecting, connect timeout the longer one */
  {BASE + 4, 0,      10000, 12000, TRUE, 6000, "6 seconds should be left"},

  };

  /* this is the pretended start time of the transfer */
  data->progress.t_startsingle.tv_sec = BASE;
  data->progress.t_startsingle.tv_usec = 0;
  data->progress.t_startop.tv_sec = BASE;
  data->progress.t_startop.tv_usec = 0;

  for(i = 0; i < sizeof(run)/sizeof(run[0]); i++) {
    timediff_t timeout;
    NOW(run[i].now_s, run[i].now_us);
    TIMEOUTS(run[i].timeout_ms, run[i].connecttimeout_ms);
    timeout =  Curl_timeleft(data, &now, run[i].connecting);
    if(timeout != run[i].result)
      fail(run[i].comment);
  }
}
UNITTEST_STOP
