/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 30, 2025.
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

/* Chirp (bandwidth expand) LP AR filter.
   This logic is reused in _celt_lpc(). Any bug fixes should also be applied there. */
void silk_bwexpander_32(
    opus_int32                  *ar,                /* I/O  AR filter to be expanded (without leading 1)                */
    const opus_int              d,                  /* I    Length of ar                                                */
    opus_int32                  chirp_Q16           /* I    Chirp factor in Q16                                         */
)
{
    opus_int   i;
    opus_int32 chirp_minus_one_Q16 = chirp_Q16 - 65536;

    for( i = 0; i < d - 1; i++ ) {
        ar[ i ]    = silk_SMULWW( chirp_Q16, ar[ i ] );
        chirp_Q16 += silk_RSHIFT_ROUND( silk_MUL( chirp_Q16, chirp_minus_one_Q16 ), 16 );
    }
    ar[ d - 1 ] = silk_SMULWW( chirp_Q16, ar[ d - 1 ] );
}

