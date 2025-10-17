/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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
 * This file contains the function WebRtcSpl_ReflCoefToLpc().
 * The description header can be found in signal_processing_library.h
 *
 */

#include "common_audio/signal_processing/include/signal_processing_library.h"

void WebRtcSpl_ReflCoefToLpc(const int16_t *k, int use_order, int16_t *a)
{
    int16_t any[WEBRTC_SPL_MAX_LPC_ORDER + 1];
    int16_t *aptr, *aptr2, *anyptr;
    const int16_t *kptr;
    int m, i;

    kptr = k;
    *a = 4096; // i.e., (Word16_MAX >> 3)+1.
    *any = *a;
    a[1] = *k >> 3;

    for (m = 1; m < use_order; m++)
    {
        kptr++;
        aptr = a;
        aptr++;
        aptr2 = &a[m];
        anyptr = any;
        anyptr++;

        any[m + 1] = *kptr >> 3;
        for (i = 0; i < m; i++)
        {
            *anyptr = *aptr + (int16_t)((*aptr2 * *kptr) >> 15);
            anyptr++;
            aptr++;
            aptr2--;
        }

        aptr = a;
        anyptr = any;
        for (i = 0; i < (m + 2); i++)
        {
            *aptr = *anyptr;
            aptr++;
            anyptr++;
        }
    }
}
