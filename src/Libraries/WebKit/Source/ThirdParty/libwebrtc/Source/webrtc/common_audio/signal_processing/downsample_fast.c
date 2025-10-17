/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
#include "common_audio/signal_processing/include/signal_processing_library.h"

#include "rtc_base/checks.h"
#include "rtc_base/sanitizer.h"

// TODO(Bjornv): Change the function parameter order to WebRTC code style.
// C version of WebRtcSpl_DownsampleFast() for generic platforms.
int WebRtcSpl_DownsampleFastC(const int16_t* data_in,
                              size_t data_in_length,
                              int16_t* data_out,
                              size_t data_out_length,
                              const int16_t* __restrict coefficients,
                              size_t coefficients_length,
                              int factor,
                              size_t delay) {
  int16_t* const original_data_out = data_out;
  size_t i = 0;
  size_t j = 0;
  int32_t out_s32 = 0;
  size_t endpos = delay + factor * (data_out_length - 1) + 1;

  // Return error if any of the running conditions doesn't meet.
  if (data_out_length == 0 || coefficients_length == 0
                           || data_in_length < endpos) {
    return -1;
  }

  rtc_MsanCheckInitialized(coefficients, sizeof(coefficients[0]),
                           coefficients_length);

  for (i = delay; i < endpos; i += factor) {
    out_s32 = 2048;  // Round value, 0.5 in Q12.

    for (j = 0; j < coefficients_length; j++) {
      // Negative overflow is permitted here, because this is
      // auto-regressive filters, and the state for each batch run is
      // stored in the "negative" positions of the output vector.
      rtc_MsanCheckInitialized(&data_in[(ptrdiff_t) i - (ptrdiff_t) j],
          sizeof(data_in[0]), 1);
      // out_s32 is in Q12 domain.
      out_s32 += coefficients[j] * data_in[(ptrdiff_t) i - (ptrdiff_t) j];
    }

    out_s32 >>= 12;  // Q0.

    // Saturate and store the output.
    *data_out++ = WebRtcSpl_SatW32ToW16(out_s32);
  }

  RTC_DCHECK_EQ(original_data_out + data_out_length, data_out);
  rtc_MsanCheckInitialized(original_data_out, sizeof(original_data_out[0]),
                           data_out_length);

  return 0;
}
