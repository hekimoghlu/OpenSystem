/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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
#include "stddef.h"

#include "rtc_base/checks.h"
#include "common_audio/signal_processing/include/signal_processing_library.h"

// TODO(bjornv): Change the return type to report errors.

void WebRtcSpl_FilterARFastQ12(const int16_t* data_in,
                               int16_t* data_out,
                               const int16_t* __restrict coefficients,
                               size_t coefficients_length,
                               size_t data_length) {
  size_t i = 0;
  size_t j = 0;

  RTC_DCHECK_GT(data_length, 0);
  RTC_DCHECK_GT(coefficients_length, 1);

  for (i = 0; i < data_length; i++) {
    int64_t output = 0;
    int64_t sum = 0;

    for (j = coefficients_length - 1; j > 0; j--) {
      // Negative overflow is permitted here, because this is
      // auto-regressive filters, and the state for each batch run is
      // stored in the "negative" positions of the output vector.
      sum += coefficients[j] * data_out[(ptrdiff_t) i - (ptrdiff_t) j];
    }

    output = coefficients[0] * data_in[i];
    output -= sum;

    // Saturate and store the output.
    output = WEBRTC_SPL_SAT(134215679, output, -134217728);
    data_out[i] = (int16_t)((output + 2048) >> 12);
  }
}
