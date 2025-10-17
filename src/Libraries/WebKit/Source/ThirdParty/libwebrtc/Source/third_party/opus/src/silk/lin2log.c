/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FIX.h"
/* Approximation of 128 * log2() (very close inverse of silk_log2lin()) */
/* Convert input to a log scale    */
opus_int32 silk_lin2log(
    const opus_int32            inLin               /* I  input in linear scale                                         */
)
{
    opus_int32 lz, frac_Q7;

    silk_CLZ_FRAC( inLin, &lz, &frac_Q7 );

    /* Piece-wise parabolic approximation */
    return silk_ADD_LSHIFT32( silk_SMLAWB( frac_Q7, silk_MUL( frac_Q7, 128 - frac_Q7 ), 179 ), 31 - lz, 7 );
}

