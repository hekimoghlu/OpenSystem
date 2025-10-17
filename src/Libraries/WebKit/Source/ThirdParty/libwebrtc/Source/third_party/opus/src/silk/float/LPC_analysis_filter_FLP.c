/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 26, 2024.
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

#include <stdlib.h>
#include "main_FLP.h"

/************************************************/
/* LPC analysis filter                          */
/* NB! State is kept internally and the         */
/* filter always starts with zero state         */
/* first Order output samples are set to zero   */
/************************************************/

/* 16th order LPC analysis filter, does not write first 16 samples */
static OPUS_INLINE void silk_LPC_analysis_filter16_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 16; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ]  * PredCoef[ 0 ]  +
                   s_ptr[ -1 ]  * PredCoef[ 1 ]  +
                   s_ptr[ -2 ]  * PredCoef[ 2 ]  +
                   s_ptr[ -3 ]  * PredCoef[ 3 ]  +
                   s_ptr[ -4 ]  * PredCoef[ 4 ]  +
                   s_ptr[ -5 ]  * PredCoef[ 5 ]  +
                   s_ptr[ -6 ]  * PredCoef[ 6 ]  +
                   s_ptr[ -7 ]  * PredCoef[ 7 ]  +
                   s_ptr[ -8 ]  * PredCoef[ 8 ]  +
                   s_ptr[ -9 ]  * PredCoef[ 9 ]  +
                   s_ptr[ -10 ] * PredCoef[ 10 ] +
                   s_ptr[ -11 ] * PredCoef[ 11 ] +
                   s_ptr[ -12 ] * PredCoef[ 12 ] +
                   s_ptr[ -13 ] * PredCoef[ 13 ] +
                   s_ptr[ -14 ] * PredCoef[ 14 ] +
                   s_ptr[ -15 ] * PredCoef[ 15 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 12th order LPC analysis filter, does not write first 12 samples */
static OPUS_INLINE void silk_LPC_analysis_filter12_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 12; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ]  * PredCoef[ 0 ]  +
                   s_ptr[ -1 ]  * PredCoef[ 1 ]  +
                   s_ptr[ -2 ]  * PredCoef[ 2 ]  +
                   s_ptr[ -3 ]  * PredCoef[ 3 ]  +
                   s_ptr[ -4 ]  * PredCoef[ 4 ]  +
                   s_ptr[ -5 ]  * PredCoef[ 5 ]  +
                   s_ptr[ -6 ]  * PredCoef[ 6 ]  +
                   s_ptr[ -7 ]  * PredCoef[ 7 ]  +
                   s_ptr[ -8 ]  * PredCoef[ 8 ]  +
                   s_ptr[ -9 ]  * PredCoef[ 9 ]  +
                   s_ptr[ -10 ] * PredCoef[ 10 ] +
                   s_ptr[ -11 ] * PredCoef[ 11 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 10th order LPC analysis filter, does not write first 10 samples */
static OPUS_INLINE void silk_LPC_analysis_filter10_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 10; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ]  +
                   s_ptr[ -6 ] * PredCoef[ 6 ]  +
                   s_ptr[ -7 ] * PredCoef[ 7 ]  +
                   s_ptr[ -8 ] * PredCoef[ 8 ]  +
                   s_ptr[ -9 ] * PredCoef[ 9 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 8th order LPC analysis filter, does not write first 8 samples */
static OPUS_INLINE void silk_LPC_analysis_filter8_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 8; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ]  +
                   s_ptr[ -6 ] * PredCoef[ 6 ]  +
                   s_ptr[ -7 ] * PredCoef[ 7 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/* 6th order LPC analysis filter, does not write first 6 samples */
static OPUS_INLINE void silk_LPC_analysis_filter6_FLP(
          silk_float                 r_LPC[],            /* O    LPC residual signal                     */
    const silk_float                 PredCoef[],         /* I    LPC coefficients                        */
    const silk_float                 s[],                /* I    Input signal                            */
    const opus_int                   length              /* I    Length of input signal                  */
)
{
    opus_int   ix;
    silk_float LPC_pred;
    const silk_float *s_ptr;

    for( ix = 6; ix < length; ix++ ) {
        s_ptr = &s[ix - 1];

        /* short-term prediction */
        LPC_pred = s_ptr[  0 ] * PredCoef[ 0 ]  +
                   s_ptr[ -1 ] * PredCoef[ 1 ]  +
                   s_ptr[ -2 ] * PredCoef[ 2 ]  +
                   s_ptr[ -3 ] * PredCoef[ 3 ]  +
                   s_ptr[ -4 ] * PredCoef[ 4 ]  +
                   s_ptr[ -5 ] * PredCoef[ 5 ];

        /* prediction error */
        r_LPC[ix] = s_ptr[ 1 ] - LPC_pred;
    }
}

/************************************************/
/* LPC analysis filter                          */
/* NB! State is kept internally and the         */
/* filter always starts with zero state         */
/* first Order output samples are set to zero   */
/************************************************/
void silk_LPC_analysis_filter_FLP(
    silk_float                      r_LPC[],                            /* O    LPC residual signal                         */
    const silk_float                PredCoef[],                         /* I    LPC coefficients                            */
    const silk_float                s[],                                /* I    Input signal                                */
    const opus_int                  length,                             /* I    Length of input signal                      */
    const opus_int                  Order                               /* I    LPC order                                   */
)
{
    celt_assert( Order <= length );

    switch( Order ) {
        case 6:
            silk_LPC_analysis_filter6_FLP(  r_LPC, PredCoef, s, length );
        break;

        case 8:
            silk_LPC_analysis_filter8_FLP(  r_LPC, PredCoef, s, length );
        break;

        case 10:
            silk_LPC_analysis_filter10_FLP( r_LPC, PredCoef, s, length );
        break;

        case 12:
            silk_LPC_analysis_filter12_FLP( r_LPC, PredCoef, s, length );
        break;

        case 16:
            silk_LPC_analysis_filter16_FLP( r_LPC, PredCoef, s, length );
        break;

        default:
            celt_assert( 0 );
        break;
    }

    /* Set first Order output samples to zero */
    silk_memset( r_LPC, 0, Order * sizeof( silk_float ) );
}

