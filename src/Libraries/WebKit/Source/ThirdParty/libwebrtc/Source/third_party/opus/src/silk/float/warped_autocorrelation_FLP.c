/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 13, 2023.
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

#include "main_FLP.h"

/* Autocorrelations for a warped frequency axis */
void silk_warped_autocorrelation_FLP(
    silk_float                      *corr,                              /* O    Result [order + 1]                          */
    const silk_float                *input,                             /* I    Input data to correlate                     */
    const silk_float                warping,                            /* I    Warping coefficient                         */
    const opus_int                  length,                             /* I    Length of input                             */
    const opus_int                  order                               /* I    Correlation order (even)                    */
)
{
    opus_int    n, i;
    double      tmp1, tmp2;
    double      state[ MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };
    double      C[     MAX_SHAPE_LPC_ORDER + 1 ] = { 0 };

    /* Order must be even */
    celt_assert( ( order & 1 ) == 0 );

    /* Loop over samples */
    for( n = 0; n < length; n++ ) {
        tmp1 = input[ n ];
        /* Loop over allpass sections */
        for( i = 0; i < order; i += 2 ) {
            /* Output of allpass section */
            tmp2 = state[ i ] + warping * ( state[ i + 1 ] - tmp1 );
            state[ i ] = tmp1;
            C[ i ] += state[ 0 ] * tmp1;
            /* Output of allpass section */
            tmp1 = state[ i + 1 ] + warping * ( state[ i + 2 ] - tmp2 );
            state[ i + 1 ] = tmp2;
            C[ i + 1 ] += state[ 0 ] * tmp2;
        }
        state[ order ] = tmp1;
        C[ order ] += state[ 0 ] * tmp1;
    }

    /* Copy correlations in silk_float output format */
    for( i = 0; i < order + 1; i++ ) {
        corr[ i ] = ( silk_float )C[ i ];
    }
}
