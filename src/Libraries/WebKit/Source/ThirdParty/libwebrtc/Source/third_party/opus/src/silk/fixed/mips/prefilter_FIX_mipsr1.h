/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
#ifndef __PREFILTER_FIX_MIPSR1_H__
#define __PREFILTER_FIX_MIPSR1_H__

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "main_FIX.h"
#include "stack_alloc.h"
#include "tuning_parameters.h"

#define OVERRIDE_silk_warped_LPC_analysis_filter_FIX
void silk_warped_LPC_analysis_filter_FIX(
          opus_int32            state[],                    /* I/O  State [order + 1]                   */
          opus_int32            res_Q2[],                   /* O    Residual signal [length]            */
    const opus_int16            coef_Q13[],                 /* I    Coefficients [order]                */
    const opus_int16            input[],                    /* I    Input signal [length]               */
    const opus_int16            lambda_Q16,                 /* I    Warping factor                      */
    const opus_int              length,                     /* I    Length of input signal              */
    const opus_int              order,                      /* I    Filter order (even)                 */
               int              arch
)
{
    opus_int     n, i;
    opus_int32   acc_Q11, acc_Q22, tmp1, tmp2, tmp3, tmp4;
    opus_int32   state_cur, state_next;

    (void)arch;

    /* Order must be even */
    /* Length must be even */

    silk_assert( ( order & 1 ) == 0 );
    silk_assert( ( length & 1 ) == 0 );

    for( n = 0; n < length; n+=2 ) {
        /* Output of lowpass section */
        tmp2 = silk_SMLAWB( state[ 0 ], state[ 1 ], lambda_Q16 );
        state_cur = silk_LSHIFT( input[ n ], 14 );
        /* Output of allpass section */
        tmp1 = silk_SMLAWB( state[ 1 ], state[ 2 ] - tmp2, lambda_Q16 );
        state_next = tmp2;
        acc_Q11 = silk_RSHIFT( order, 1 );
        acc_Q11 = silk_SMLAWB( acc_Q11, tmp2, coef_Q13[ 0 ] );


        /* Output of lowpass section */
        tmp4 = silk_SMLAWB( state_cur, state_next, lambda_Q16 );
        state[ 0 ] = silk_LSHIFT( input[ n+1 ], 14 );
        /* Output of allpass section */
        tmp3 = silk_SMLAWB( state_next, tmp1 - tmp4, lambda_Q16 );
        state[ 1 ] = tmp4;
        acc_Q22 = silk_RSHIFT( order, 1 );
        acc_Q22 = silk_SMLAWB( acc_Q22, tmp4, coef_Q13[ 0 ] );

        /* Loop over allpass sections */
        for( i = 2; i < order; i += 2 ) {
            /* Output of allpass section */
            tmp2 = silk_SMLAWB( state[ i ], state[ i + 1 ] - tmp1, lambda_Q16 );
            state_cur = tmp1;
            acc_Q11 = silk_SMLAWB( acc_Q11, tmp1, coef_Q13[ i - 1 ] );
            /* Output of allpass section */
            tmp1 = silk_SMLAWB( state[ i + 1 ], state[ i + 2 ] - tmp2, lambda_Q16 );
            state_next = tmp2;
            acc_Q11 = silk_SMLAWB( acc_Q11, tmp2, coef_Q13[ i ] );


            /* Output of allpass section */
            tmp4 = silk_SMLAWB( state_cur, state_next - tmp3, lambda_Q16 );
            state[ i ] = tmp3;
            acc_Q22 = silk_SMLAWB( acc_Q22, tmp3, coef_Q13[ i - 1 ] );
            /* Output of allpass section */
            tmp3 = silk_SMLAWB( state_next, tmp1 - tmp4, lambda_Q16 );
            state[ i + 1 ] = tmp4;
            acc_Q22 = silk_SMLAWB( acc_Q22, tmp4, coef_Q13[ i ] );
        }
        acc_Q11 = silk_SMLAWB( acc_Q11, tmp1, coef_Q13[ order - 1 ] );
        res_Q2[ n ] = silk_LSHIFT( (opus_int32)input[ n ], 2 ) - silk_RSHIFT_ROUND( acc_Q11, 9 );

        state[ order ] = tmp3;
        acc_Q22 = silk_SMLAWB( acc_Q22, tmp3, coef_Q13[ order - 1 ] );
        res_Q2[ n+1 ] = silk_LSHIFT( (opus_int32)input[ n+1 ], 2 ) - silk_RSHIFT_ROUND( acc_Q22, 9 );
    }
}



/* Prefilter for finding Quantizer input signal */
#define OVERRIDE_silk_prefilt_FIX
static inline void silk_prefilt_FIX(
    silk_prefilter_state_FIX    *P,                         /* I/O  state                               */
    opus_int32                  st_res_Q12[],               /* I    short term residual signal          */
    opus_int32                  xw_Q3[],                    /* O    prefiltered signal                  */
    opus_int32                  HarmShapeFIRPacked_Q12,     /* I    Harmonic shaping coeficients        */
    opus_int                    Tilt_Q14,                   /* I    Tilt shaping coeficient             */
    opus_int32                  LF_shp_Q14,                 /* I    Low-frequancy shaping coeficients   */
    opus_int                    lag,                        /* I    Lag for harmonic shaping            */
    opus_int                    length                      /* I    Length of signals                   */
)
{
    opus_int   i, idx, LTP_shp_buf_idx;
    opus_int32 n_LTP_Q12, n_Tilt_Q10, n_LF_Q10;
    opus_int32 sLF_MA_shp_Q12, sLF_AR_shp_Q12;
    opus_int16 *LTP_shp_buf;

    /* To speed up use temp variables instead of using the struct */
    LTP_shp_buf     = P->sLTP_shp;
    LTP_shp_buf_idx = P->sLTP_shp_buf_idx;
    sLF_AR_shp_Q12  = P->sLF_AR_shp_Q12;
    sLF_MA_shp_Q12  = P->sLF_MA_shp_Q12;

    if( lag > 0 ) {
        for( i = 0; i < length; i++ ) {
            /* unrolled loop */
            silk_assert( HARM_SHAPE_FIR_TAPS == 3 );
            idx = lag + LTP_shp_buf_idx;
            n_LTP_Q12 = silk_SMULBB(            LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2 - 1) & LTP_MASK ], HarmShapeFIRPacked_Q12 );
            n_LTP_Q12 = silk_SMLABT( n_LTP_Q12, LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2    ) & LTP_MASK ], HarmShapeFIRPacked_Q12 );
            n_LTP_Q12 = silk_SMLABB( n_LTP_Q12, LTP_shp_buf[ ( idx - HARM_SHAPE_FIR_TAPS / 2 + 1) & LTP_MASK ], HarmShapeFIRPacked_Q12 );

            n_Tilt_Q10 = silk_SMULWB( sLF_AR_shp_Q12, Tilt_Q14 );
            n_LF_Q10   = silk_SMLAWB( silk_SMULWT( sLF_AR_shp_Q12, LF_shp_Q14 ), sLF_MA_shp_Q12, LF_shp_Q14 );

            sLF_AR_shp_Q12 = silk_SUB32( st_res_Q12[ i ], silk_LSHIFT( n_Tilt_Q10, 2 ) );
            sLF_MA_shp_Q12 = silk_SUB32( sLF_AR_shp_Q12,  silk_LSHIFT( n_LF_Q10,   2 ) );

            LTP_shp_buf_idx = ( LTP_shp_buf_idx - 1 ) & LTP_MASK;
            LTP_shp_buf[ LTP_shp_buf_idx ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( sLF_MA_shp_Q12, 12 ) );

            xw_Q3[i] = silk_RSHIFT_ROUND( silk_SUB32( sLF_MA_shp_Q12, n_LTP_Q12 ), 9 );
        }
    }
    else
    {
        for( i = 0; i < length; i++ ) {

            n_LTP_Q12 = 0;

            n_Tilt_Q10 = silk_SMULWB( sLF_AR_shp_Q12, Tilt_Q14 );
            n_LF_Q10   = silk_SMLAWB( silk_SMULWT( sLF_AR_shp_Q12, LF_shp_Q14 ), sLF_MA_shp_Q12, LF_shp_Q14 );

            sLF_AR_shp_Q12 = silk_SUB32( st_res_Q12[ i ], silk_LSHIFT( n_Tilt_Q10, 2 ) );
            sLF_MA_shp_Q12 = silk_SUB32( sLF_AR_shp_Q12,  silk_LSHIFT( n_LF_Q10,   2 ) );

            LTP_shp_buf_idx = ( LTP_shp_buf_idx - 1 ) & LTP_MASK;
            LTP_shp_buf[ LTP_shp_buf_idx ] = (opus_int16)silk_SAT16( silk_RSHIFT_ROUND( sLF_MA_shp_Q12, 12 ) );

            xw_Q3[i] = silk_RSHIFT_ROUND( sLF_MA_shp_Q12, 9 );
        }
    }

    /* Copy temp variable back to state */
    P->sLF_AR_shp_Q12   = sLF_AR_shp_Q12;
    P->sLF_MA_shp_Q12   = sLF_MA_shp_Q12;
    P->sLTP_shp_buf_idx = LTP_shp_buf_idx;
}

#endif /* __PREFILTER_FIX_MIPSR1_H__ */
