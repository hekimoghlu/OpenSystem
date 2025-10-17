/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#include "curl_setup.h"

#include <curl/curl.h>
#include "urldata.h"
#include "sendf.h"
#include "multiif.h"
#include "speedcheck.h"

void Curl_speedinit(struct Curl_easy *data)
{
  memset(&data->state.keeps_speed, 0, sizeof(struct curltime));
}

/*
 * @unittest: 1606
 */
CURLcode Curl_speedcheck(struct Curl_easy *data,
                         struct curltime now)
{
  if(data->req.keepon & KEEP_RECV_PAUSE)
    /* A paused transfer is not qualified for speed checks */
    return CURLE_OK;

  if((data->progress.current_speed >= 0) && data->set.low_speed_time) {
    if(data->progress.current_speed < data->set.low_speed_limit) {
      if(!data->state.keeps_speed.tv_sec)
        /* under the limit at this very moment */
        data->state.keeps_speed = now;
      else {
        /* how long has it been under the limit */
        timediff_t howlong = Curl_timediff(now, data->state.keeps_speed);

        if(howlong >= data->set.low_speed_time * 1000) {
          /* too long */
          failf(data,
                "Operation too slow. "
                "Less than %ld bytes/sec transferred the last %ld seconds",
                data->set.low_speed_limit,
                data->set.low_speed_time);
          return CURLE_OPERATION_TIMEDOUT;
        }
      }
    }
    else
      /* faster right now */
      data->state.keeps_speed.tv_sec = 0;
  }

  if(data->set.low_speed_limit)
    /* if low speed limit is enabled, set the expire timer to make this
       connection's speed get checked again in a second */
    Curl_expire(data, 1000, EXPIRE_SPEEDCHECK);

  return CURLE_OK;
}

