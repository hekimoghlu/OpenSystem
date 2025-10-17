/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

#include "define.h"
#include "main_FLP.h"
#include "tuning_parameters.h"

/* LPC analysis */
void silk_find_LPC_FLP(
    silk_encoder_state              *psEncC,                            /* I/O  Encoder state                               */
    opus_int16                      NLSF_Q15[],                         /* O    NLSFs                                       */
    const silk_float                x[],                                /* I    Input signal                                */
    const silk_float                minInvGain                          /* I    Inverse of max prediction gain              */
)
{
    opus_int    k, subfr_length;
    silk_float  a[ MAX_LPC_ORDER ];

    /* Used only for NLSF interpolation */
    silk_float  res_nrg, res_nrg_2nd, res_nrg_interp;
    opus_int16  NLSF0_Q15[ MAX_LPC_ORDER ];
    silk_float  a_tmp[ MAX_LPC_ORDER ];
    silk_float  LPC_res[ MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER ];

    subfr_length = psEncC->subfr_length + psEncC->predictLPCOrder;

    /* Default: No interpolation */
    psEncC->indices.NLSFInterpCoef_Q2 = 4;

    /* Burg AR analysis for the full frame */
    res_nrg = silk_burg_modified_FLP( a, x, minInvGain, subfr_length, psEncC->nb_subfr, psEncC->predictLPCOrder );

    if( psEncC->useInterpolatedNLSFs && !psEncC->first_frame_after_reset && psEncC->nb_subfr == MAX_NB_SUBFR ) {
        /* Optimal solution for last 10 ms; subtract residual energy here, as that's easier than        */
        /* adding it to the residual energy of the first 10 ms in each iteration of the search below    */
        res_nrg -= silk_burg_modified_FLP( a_tmp, x + ( MAX_NB_SUBFR / 2 ) * subfr_length, minInvGain, subfr_length, MAX_NB_SUBFR / 2, psEncC->predictLPCOrder );

        /* Convert to NLSFs */
        silk_A2NLSF_FLP( NLSF_Q15, a_tmp, psEncC->predictLPCOrder );

        /* Search over interpolation indices to find the one with lowest residual energy */
        res_nrg_2nd = silk_float_MAX;
        for( k = 3; k >= 0; k-- ) {
            /* Interpolate NLSFs for first half */
            silk_interpolate( NLSF0_Q15, psEncC->prev_NLSFq_Q15, NLSF_Q15, k, psEncC->predictLPCOrder );

            /* Convert to LPC for residual energy evaluation */
            silk_NLSF2A_FLP( a_tmp, NLSF0_Q15, psEncC->predictLPCOrder, psEncC->arch );

            /* Calculate residual energy with LSF interpolation */
            silk_LPC_analysis_filter_FLP( LPC_res, a_tmp, x, 2 * subfr_length, psEncC->predictLPCOrder );
            res_nrg_interp = (silk_float)(
                silk_energy_FLP( LPC_res + psEncC->predictLPCOrder,                subfr_length - psEncC->predictLPCOrder ) +
                silk_energy_FLP( LPC_res + psEncC->predictLPCOrder + subfr_length, subfr_length - psEncC->predictLPCOrder ) );

            /* Determine whether current interpolated NLSFs are best so far */
            if( res_nrg_interp < res_nrg ) {
                /* Interpolation has lower residual energy */
                res_nrg = res_nrg_interp;
                psEncC->indices.NLSFInterpCoef_Q2 = (opus_int8)k;
            } else if( res_nrg_interp > res_nrg_2nd ) {
                /* No reason to continue iterating - residual energies will continue to climb */
                break;
            }
            res_nrg_2nd = res_nrg_interp;
        }
    }

    if( psEncC->indices.NLSFInterpCoef_Q2 == 4 ) {
        /* NLSF interpolation is currently inactive, calculate NLSFs from full frame AR coefficients */
        silk_A2NLSF_FLP( NLSF_Q15, a, psEncC->predictLPCOrder );
    }

    celt_assert( psEncC->indices.NLSFInterpCoef_Q2 == 4 ||
        ( psEncC->useInterpolatedNLSFs && !psEncC->first_frame_after_reset && psEncC->nb_subfr == MAX_NB_SUBFR ) );
}
