/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

/* Compute number of bits to right shift the sum of squares of a vector */
/* of int16s to make it fit in an int32                                 */
void silk_sum_sqr_shift(
    opus_int32                  *energy,            /* O   Energy of x, after shifting to the right                     */
    opus_int                    *shift,             /* O   Number of bits right shift applied to energy                 */
    const opus_int16            *x,                 /* I   Input vector                                                 */
    opus_int                    len                 /* I   Length of input vector                                       */
)
{
    opus_int   i, shft;
    opus_uint32 nrg_tmp;
    opus_int32 nrg;

    /* Do a first run with the maximum shift we could have. */
    shft = 31-silk_CLZ32(len);
    /* Let's be conservative with rounding and start with nrg=len. */
    nrg  = len;
    for( i = 0; i < len - 1; i += 2 ) {
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg_tmp = silk_SMLABB_ovflw( nrg_tmp, x[ i + 1 ], x[ i + 1 ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }
    if( i < len ) {
        /* One sample left to process */
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }
    silk_assert( nrg >= 0 );
    /* Make sure the result will fit in a 32-bit signed integer with two bits
       of headroom. */
    shft = silk_max_32(0, shft+3 - silk_CLZ32(nrg));
    nrg = 0;
    for( i = 0 ; i < len - 1; i += 2 ) {
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg_tmp = silk_SMLABB_ovflw( nrg_tmp, x[ i + 1 ], x[ i + 1 ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }
    if( i < len ) {
        /* One sample left to process */
        nrg_tmp = silk_SMULBB( x[ i ], x[ i ] );
        nrg = (opus_int32)silk_ADD_RSHIFT_uint( nrg, nrg_tmp, shft );
    }

    silk_assert( nrg >= 0 );

    /* Output arguments */
    *shift  = shft;
    *energy = nrg;
}

