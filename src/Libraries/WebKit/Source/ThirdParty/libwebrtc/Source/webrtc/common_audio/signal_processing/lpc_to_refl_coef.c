/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 23, 2024.
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
 * This file contains the function WebRtcSpl_LpcToReflCoef().
 * The description header can be found in signal_processing_library.h
 *
 */

#include "common_audio/signal_processing/include/signal_processing_library.h"

#define SPL_LPC_TO_REFL_COEF_MAX_AR_MODEL_ORDER 50

void WebRtcSpl_LpcToReflCoef(int16_t* a16, int use_order, int16_t* k16)
{
    int m, k;
    int32_t tmp32[SPL_LPC_TO_REFL_COEF_MAX_AR_MODEL_ORDER];
    int32_t tmp_inv_denom32;
    int16_t tmp_inv_denom16;

    k16[use_order - 1] = a16[use_order] << 3;  // Q12<<3 => Q15
    for (m = use_order - 1; m > 0; m--)
    {
        // (1 - k^2) in Q30
        tmp_inv_denom32 = 1073741823 - k16[m] * k16[m];
        // (1 - k^2) in Q15
        tmp_inv_denom16 = (int16_t)(tmp_inv_denom32 >> 15);

        for (k = 1; k <= m; k++)
        {
            // tmp[k] = (a[k] - RC[m] * a[m-k+1]) / (1.0 - RC[m]*RC[m]);

            // [Q12<<16 - (Q15*Q12)<<1] = [Q28 - Q28] = Q28
            tmp32[k] = (a16[k] << 16) - (k16[m] * a16[m - k + 1] << 1);

            tmp32[k] = WebRtcSpl_DivW32W16(tmp32[k], tmp_inv_denom16); //Q28/Q15 = Q13
        }

        for (k = 1; k < m; k++)
        {
            a16[k] = (int16_t)(tmp32[k] >> 1);  // Q13>>1 => Q12
        }

        tmp32[m] = WEBRTC_SPL_SAT(8191, tmp32[m], -8191);
        k16[m - 1] = (int16_t)WEBRTC_SPL_LSHIFT_W32(tmp32[m], 2); //Q13<<2 => Q15
    }
    return;
}
