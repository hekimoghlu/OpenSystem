/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#include "curl_range.h"
#include "sendf.h"
#include "strtoofft.h"

/* Only include this function if one or more of FTP, FILE are enabled. */
#if !defined(CURL_DISABLE_FTP) || !defined(CURL_DISABLE_FILE)

 /*
  Check if this is a range download, and if so, set the internal variables
  properly.
 */
CURLcode Curl_range(struct Curl_easy *data)
{
  curl_off_t from, to;
  char *ptr;
  char *ptr2;

  if(data->state.use_range && data->state.range) {
    CURLofft from_t;
    CURLofft to_t;
    from_t = curlx_strtoofft(data->state.range, &ptr, 10, &from);
    if(from_t == CURL_OFFT_FLOW)
      return CURLE_RANGE_ERROR;
    while(*ptr && (ISBLANK(*ptr) || (*ptr == '-')))
      ptr++;
    to_t = curlx_strtoofft(ptr, &ptr2, 10, &to);
    if(to_t == CURL_OFFT_FLOW)
      return CURLE_RANGE_ERROR;
    if((to_t == CURL_OFFT_INVAL) && !from_t) {
      /* X - */
      data->state.resume_from = from;
      DEBUGF(infof(data, "RANGE %" CURL_FORMAT_CURL_OFF_T " to end of file",
                   from));
    }
    else if((from_t == CURL_OFFT_INVAL) && !to_t) {
      /* -Y */
      data->req.maxdownload = to;
      data->state.resume_from = -to;
      DEBUGF(infof(data, "RANGE the last %" CURL_FORMAT_CURL_OFF_T " bytes",
                   to));
    }
    else {
      /* X-Y */
      curl_off_t totalsize;

      /* Ensure the range is sensible - to should follow from. */
      if(from > to)
        return CURLE_RANGE_ERROR;

      totalsize = to - from;
      if(totalsize == CURL_OFF_T_MAX)
        return CURLE_RANGE_ERROR;

      data->req.maxdownload = totalsize + 1; /* include last byte */
      data->state.resume_from = from;
      DEBUGF(infof(data, "RANGE from %" CURL_FORMAT_CURL_OFF_T
                   " getting %" CURL_FORMAT_CURL_OFF_T " bytes",
                   from, data->req.maxdownload));
    }
    DEBUGF(infof(data, "range-download from %" CURL_FORMAT_CURL_OFF_T
                 " to %" CURL_FORMAT_CURL_OFF_T ", totally %"
                 CURL_FORMAT_CURL_OFF_T " bytes",
                 from, to, data->req.maxdownload));
  }
  else
    data->req.maxdownload = -1;
  return CURLE_OK;
}

#endif
