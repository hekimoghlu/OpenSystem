/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
 * This file contains the function WebRtcSpl_Energy().
 * The description header can be found in signal_processing_library.h
 *
 */

#include "common_audio/signal_processing/include/signal_processing_library.h"

int32_t WebRtcSpl_Energy(int16_t* vector,
                         size_t vector_length,
                         int* scale_factor)
{
    int32_t en = 0;
    size_t i;
    int scaling =
        WebRtcSpl_GetScalingSquare(vector, vector_length, vector_length);
    size_t looptimes = vector_length;
    int16_t *vectorptr = vector;

    for (i = 0; i < looptimes; i++)
    {
      en += (*vectorptr * *vectorptr) >> scaling;
      vectorptr++;
    }
    *scale_factor = scaling;

    return en;
}
