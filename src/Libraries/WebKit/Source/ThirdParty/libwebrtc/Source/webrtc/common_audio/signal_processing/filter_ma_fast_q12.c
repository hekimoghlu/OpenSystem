/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
/*
 * This file contains the function WebRtcSpl_FilterMAFastQ12().
 * The description header can be found in signal_processing_library.h
 *
 */

#include "common_audio/signal_processing/include/signal_processing_library.h"

#include "rtc_base/sanitizer.h"

void WebRtcSpl_FilterMAFastQ12(const int16_t* in_ptr,
                               int16_t* out_ptr,
                               const int16_t* B,
                               size_t B_length,
                               size_t length)
{
    size_t i, j;

    rtc_MsanCheckInitialized(B, sizeof(B[0]), B_length);
    rtc_MsanCheckInitialized(in_ptr - B_length + 1, sizeof(in_ptr[0]),
                             B_length + length - 1);

    for (i = 0; i < length; i++)
    {
        int32_t o = 0;

        for (j = 0; j < B_length; j++)
        {
          // Negative overflow is permitted here, because this is
          // auto-regressive filters, and the state for each batch run is
          // stored in the "negative" positions of the output vector.
          o += B[j] * in_ptr[(ptrdiff_t) i - (ptrdiff_t) j];
        }

        // If output is higher than 32768, saturate it. Same with negative side
        // 2^27 = 134217728, which corresponds to 32768 in Q12

        // Saturate the output
        o = WEBRTC_SPL_SAT((int32_t)134215679, o, (int32_t)-134217728);

        *out_ptr++ = (int16_t)((o + (int32_t)2048) >> 12);
    }
    return;
}
