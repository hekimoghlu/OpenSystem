/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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

#include "main.h"

/* Entropy constrained matrix-weighted VQ, hard-coded to 5-element vectors, for a single input data vector */
void silk_VQ_WMat_EC_c(
    opus_int8                   *ind,                           /* O    index of best codebook vector               */
    opus_int32                  *res_nrg_Q15,                   /* O    best residual energy                        */
    opus_int32                  *rate_dist_Q8,                  /* O    best total bitrate                          */
    opus_int                    *gain_Q7,                       /* O    sum of absolute LTP coefficients            */
    const opus_int32            *XX_Q17,                        /* I    correlation matrix                          */
    const opus_int32            *xX_Q17,                        /* I    correlation vector                          */
    const opus_int8             *cb_Q7,                         /* I    codebook                                    */
    const opus_uint8            *cb_gain_Q7,                    /* I    codebook effective gain                     */
    const opus_uint8            *cl_Q5,                         /* I    code length for each codebook vector        */
    const opus_int              subfr_len,                      /* I    number of samples per subframe              */
    const opus_int32            max_gain_Q7,                    /* I    maximum sum of absolute LTP coefficients    */
    const opus_int              L                               /* I    number of vectors in codebook               */
)
{
    opus_int   k, gain_tmp_Q7;
    const opus_int8 *cb_row_Q7;
    opus_int32 neg_xX_Q24[ 5 ];
    opus_int32 sum1_Q15, sum2_Q24;
    opus_int32 bits_res_Q8, bits_tot_Q8;

    /* Negate and convert to new Q domain */
    neg_xX_Q24[ 0 ] = -silk_LSHIFT32( xX_Q17[ 0 ], 7 );
    neg_xX_Q24[ 1 ] = -silk_LSHIFT32( xX_Q17[ 1 ], 7 );
    neg_xX_Q24[ 2 ] = -silk_LSHIFT32( xX_Q17[ 2 ], 7 );
    neg_xX_Q24[ 3 ] = -silk_LSHIFT32( xX_Q17[ 3 ], 7 );
    neg_xX_Q24[ 4 ] = -silk_LSHIFT32( xX_Q17[ 4 ], 7 );

    /* Loop over codebook */
    *rate_dist_Q8 = silk_int32_MAX;
    *res_nrg_Q15 = silk_int32_MAX;
    cb_row_Q7 = cb_Q7;
    /* If things go really bad, at least *ind is set to something safe. */
    *ind = 0;
    for( k = 0; k < L; k++ ) {
        opus_int32 penalty;
        gain_tmp_Q7 = cb_gain_Q7[k];
        /* Weighted rate */
        /* Quantization error: 1 - 2 * xX * cb + cb' * XX * cb */
        sum1_Q15 = SILK_FIX_CONST( 1.001, 15 );

        /* Penalty for too large gain */
        penalty = silk_LSHIFT32( silk_max( silk_SUB32( gain_tmp_Q7, max_gain_Q7 ), 0 ), 11 );

        /* first row of XX_Q17 */
        sum2_Q24 = silk_MLA( neg_xX_Q24[ 0 ], XX_Q17[  1 ], cb_row_Q7[ 1 ] );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  2 ], cb_row_Q7[ 2 ] );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  3 ], cb_row_Q7[ 3 ] );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  4 ], cb_row_Q7[ 4 ] );
        sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  0 ], cb_row_Q7[ 0 ] );
        sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 0 ] );

        /* second row of XX_Q17 */
        sum2_Q24 = silk_MLA( neg_xX_Q24[ 1 ], XX_Q17[  7 ], cb_row_Q7[ 2 ] );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  8 ], cb_row_Q7[ 3 ] );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  9 ], cb_row_Q7[ 4 ] );
        sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[  6 ], cb_row_Q7[ 1 ] );
        sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 1 ] );

        /* third row of XX_Q17 */
        sum2_Q24 = silk_MLA( neg_xX_Q24[ 2 ], XX_Q17[ 13 ], cb_row_Q7[ 3 ] );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 14 ], cb_row_Q7[ 4 ] );
        sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 12 ], cb_row_Q7[ 2 ] );
        sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 2 ] );

        /* fourth row of XX_Q17 */
        sum2_Q24 = silk_MLA( neg_xX_Q24[ 3 ], XX_Q17[ 19 ], cb_row_Q7[ 4 ] );
        sum2_Q24 = silk_LSHIFT32( sum2_Q24, 1 );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 18 ], cb_row_Q7[ 3 ] );
        sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 3 ] );

        /* last row of XX_Q17 */
        sum2_Q24 = silk_LSHIFT32( neg_xX_Q24[ 4 ], 1 );
        sum2_Q24 = silk_MLA( sum2_Q24,        XX_Q17[ 24 ], cb_row_Q7[ 4 ] );
        sum1_Q15 = silk_SMLAWB( sum1_Q15,        sum2_Q24,  cb_row_Q7[ 4 ] );

        /* find best */
        if( sum1_Q15 >= 0 ) {
            /* Translate residual energy to bits using high-rate assumption (6 dB ==> 1 bit/sample) */
            bits_res_Q8 = silk_SMULBB( subfr_len, silk_lin2log( sum1_Q15 + penalty) - (15 << 7) );
            /* In the following line we reduce the codelength component by half ("-1"); seems to slightly improve quality */
            bits_tot_Q8 = silk_ADD_LSHIFT32( bits_res_Q8, cl_Q5[ k ], 3-1 );
            if( bits_tot_Q8 <= *rate_dist_Q8 ) {
                *rate_dist_Q8 = bits_tot_Q8;
                *res_nrg_Q15 = sum1_Q15 + penalty;
                *ind = (opus_int8)k;
                *gain_Q7 = gain_tmp_Q7;
            }
        }

        /* Go to next cbk vector */
        cb_row_Q7 += LTP_ORDER;
    }
}
