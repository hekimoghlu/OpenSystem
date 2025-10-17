/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 8, 2025.
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
 * This file contains the function WebRtcSpl_SqrtOfOneMinusXSquared().
 * The description header can be found in signal_processing_library.h
 *
 */

#include "common_audio/signal_processing/include/signal_processing_library.h"

void WebRtcSpl_SqrtOfOneMinusXSquared(int16_t *xQ15, size_t vector_length,
                                      int16_t *yQ15)
{
    int32_t sq;
    size_t m;
    int16_t tmp;

    for (m = 0; m < vector_length; m++)
    {
        tmp = xQ15[m];
        sq = tmp * tmp;  // x^2 in Q30
        sq = 1073741823 - sq; // 1-x^2, where 1 ~= 0.99999999906 is 1073741823 in Q30
        sq = WebRtcSpl_Sqrt(sq); // sqrt(1-x^2) in Q15
        yQ15[m] = (int16_t)sq;
    }
}
