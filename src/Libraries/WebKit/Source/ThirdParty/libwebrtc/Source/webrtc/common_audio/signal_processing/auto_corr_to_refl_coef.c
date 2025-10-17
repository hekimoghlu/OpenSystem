/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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
 * This file contains the function WebRtcSpl_AutoCorrToReflCoef().
 * The description header can be found in signal_processing_library.h
 *
 */

#include "common_audio/signal_processing/include/signal_processing_library.h"

void WebRtcSpl_AutoCorrToReflCoef(const int32_t *R, int use_order, int16_t *K)
{
    int i, n;
    int16_t tmp;
    const int32_t *rptr;
    int32_t L_num, L_den;
    int16_t *acfptr, *pptr, *wptr, *p1ptr, *w1ptr, ACF[WEBRTC_SPL_MAX_LPC_ORDER],
            P[WEBRTC_SPL_MAX_LPC_ORDER], W[WEBRTC_SPL_MAX_LPC_ORDER];

    // Initialize loop and pointers.
    acfptr = ACF;
    rptr = R;
    pptr = P;
    p1ptr = &P[1];
    w1ptr = &W[1];
    wptr = w1ptr;

    // First loop; n=0. Determine shifting.
    tmp = WebRtcSpl_NormW32(*R);
    *acfptr = (int16_t)((*rptr++ << tmp) >> 16);
    *pptr++ = *acfptr++;

    // Initialize ACF, P and W.
    for (i = 1; i <= use_order; i++)
    {
        *acfptr = (int16_t)((*rptr++ << tmp) >> 16);
        *wptr++ = *acfptr;
        *pptr++ = *acfptr++;
    }

    // Compute reflection coefficients.
    for (n = 1; n <= use_order; n++, K++)
    {
        tmp = WEBRTC_SPL_ABS_W16(*p1ptr);
        if (*P < tmp)
        {
            for (i = n; i <= use_order; i++)
                *K++ = 0;

            return;
        }

        // Division: WebRtcSpl_div(tmp, *P)
        *K = 0;
        if (tmp != 0)
        {
            L_num = tmp;
            L_den = *P;
            i = 15;
            while (i--)
            {
                (*K) <<= 1;
                L_num <<= 1;
                if (L_num >= L_den)
                {
                    L_num -= L_den;
                    (*K)++;
                }
            }
            if (*p1ptr > 0)
                *K = -*K;
        }

        // Last iteration; don't do Schur recursion.
        if (n == use_order)
            return;

        // Schur recursion.
        pptr = P;
        wptr = w1ptr;
        tmp = (int16_t)(((int32_t)*p1ptr * (int32_t)*K + 16384) >> 15);
        *pptr = WebRtcSpl_AddSatW16(*pptr, tmp);
        pptr++;
        for (i = 1; i <= use_order - n; i++)
        {
            tmp = (int16_t)(((int32_t)*wptr * (int32_t)*K + 16384) >> 15);
            *pptr = WebRtcSpl_AddSatW16(*(pptr + 1), tmp);
            pptr++;
            tmp = (int16_t)(((int32_t)*pptr * (int32_t)*K + 16384) >> 15);
            *wptr = WebRtcSpl_AddSatW16(*wptr, tmp);
            wptr++;
        }
    }
}
