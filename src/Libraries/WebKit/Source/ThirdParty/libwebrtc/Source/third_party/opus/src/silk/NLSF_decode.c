/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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

/* Predictive dequantizer for NLSF residuals */
static OPUS_INLINE void silk_NLSF_residual_dequant(          /* O    Returns RD value in Q30                     */
          opus_int16         x_Q10[],                        /* O    Output [ order ]                            */
    const opus_int8          indices[],                      /* I    Quantization indices [ order ]              */
    const opus_uint8         pred_coef_Q8[],                 /* I    Backward predictor coefs [ order ]          */
    const opus_int           quant_step_size_Q16,            /* I    Quantization step size                      */
    const opus_int16         order                           /* I    Number of input values                      */
)
{
    opus_int     i, out_Q10, pred_Q10;

    out_Q10 = 0;
    for( i = order-1; i >= 0; i-- ) {
        pred_Q10 = silk_RSHIFT( silk_SMULBB( out_Q10, (opus_int16)pred_coef_Q8[ i ] ), 8 );
        out_Q10  = silk_LSHIFT( indices[ i ], 10 );
        if( out_Q10 > 0 ) {
            out_Q10 = silk_SUB16( out_Q10, SILK_FIX_CONST( NLSF_QUANT_LEVEL_ADJ, 10 ) );
        } else if( out_Q10 < 0 ) {
            out_Q10 = silk_ADD16( out_Q10, SILK_FIX_CONST( NLSF_QUANT_LEVEL_ADJ, 10 ) );
        }
        out_Q10  = silk_SMLAWB( pred_Q10, (opus_int32)out_Q10, quant_step_size_Q16 );
        x_Q10[ i ] = out_Q10;
    }
}


/***********************/
/* NLSF vector decoder */
/***********************/
void silk_NLSF_decode(
          opus_int16            *pNLSF_Q15,                     /* O    Quantized NLSF vector [ LPC_ORDER ]         */
          opus_int8             *NLSFIndices,                   /* I    Codebook path vector [ LPC_ORDER + 1 ]      */
    const silk_NLSF_CB_struct   *psNLSF_CB                      /* I    Codebook object                             */
)
{
    opus_int         i;
    opus_uint8       pred_Q8[  MAX_LPC_ORDER ];
    opus_int16       ec_ix[    MAX_LPC_ORDER ];
    opus_int16       res_Q10[  MAX_LPC_ORDER ];
    opus_int32       NLSF_Q15_tmp;
    const opus_uint8 *pCB_element;
    const opus_int16 *pCB_Wght_Q9;

    /* Unpack entropy table indices and predictor for current CB1 index */
    silk_NLSF_unpack( ec_ix, pred_Q8, psNLSF_CB, NLSFIndices[ 0 ] );

    /* Predictive residual dequantizer */
    silk_NLSF_residual_dequant( res_Q10, &NLSFIndices[ 1 ], pred_Q8, psNLSF_CB->quantStepSize_Q16, psNLSF_CB->order );

    /* Apply inverse square-rooted weights to first stage and add to output */
    pCB_element = &psNLSF_CB->CB1_NLSF_Q8[ NLSFIndices[ 0 ] * psNLSF_CB->order ];
    pCB_Wght_Q9 = &psNLSF_CB->CB1_Wght_Q9[ NLSFIndices[ 0 ] * psNLSF_CB->order ];
    for( i = 0; i < psNLSF_CB->order; i++ ) {
        NLSF_Q15_tmp = silk_ADD_LSHIFT32( silk_DIV32_16( silk_LSHIFT( (opus_int32)res_Q10[ i ], 14 ), pCB_Wght_Q9[ i ] ), (opus_int16)pCB_element[ i ], 7 );
        pNLSF_Q15[ i ] = (opus_int16)silk_LIMIT( NLSF_Q15_tmp, 0, 32767 );
    }

    /* NLSF stabilization */
    silk_NLSF_stabilize( pNLSF_Q15, psNLSF_CB->deltaMin_Q15, psNLSF_CB->order );
}
