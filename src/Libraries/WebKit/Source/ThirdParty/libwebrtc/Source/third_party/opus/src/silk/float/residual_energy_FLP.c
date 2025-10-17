/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

#define MAX_ITERATIONS_RESIDUAL_NRG         10
#define REGULARIZATION_FACTOR               1e-8f

/* Residual energy: nrg = wxx - 2 * wXx * c + c' * wXX * c */
silk_float silk_residual_energy_covar_FLP(                              /* O    Weighted residual energy                    */
    const silk_float                *c,                                 /* I    Filter coefficients                         */
    silk_float                      *wXX,                               /* I/O  Weighted correlation matrix, reg. out       */
    const silk_float                *wXx,                               /* I    Weighted correlation vector                 */
    const silk_float                wxx,                                /* I    Weighted correlation value                  */
    const opus_int                  D                                   /* I    Dimension                                   */
)
{
    opus_int   i, j, k;
    silk_float tmp, nrg = 0.0f, regularization;

    /* Safety checks */
    celt_assert( D >= 0 );

    regularization = REGULARIZATION_FACTOR * ( wXX[ 0 ] + wXX[ D * D - 1 ] );
    for( k = 0; k < MAX_ITERATIONS_RESIDUAL_NRG; k++ ) {
        nrg = wxx;

        tmp = 0.0f;
        for( i = 0; i < D; i++ ) {
            tmp += wXx[ i ] * c[ i ];
        }
        nrg -= 2.0f * tmp;

        /* compute c' * wXX * c, assuming wXX is symmetric */
        for( i = 0; i < D; i++ ) {
            tmp = 0.0f;
            for( j = i + 1; j < D; j++ ) {
                tmp += matrix_c_ptr( wXX, i, j, D ) * c[ j ];
            }
            nrg += c[ i ] * ( 2.0f * tmp + matrix_c_ptr( wXX, i, i, D ) * c[ i ] );
        }
        if( nrg > 0 ) {
            break;
        } else {
            /* Add white noise */
            for( i = 0; i < D; i++ ) {
                matrix_c_ptr( wXX, i, i, D ) +=  regularization;
            }
            /* Increase noise for next run */
            regularization *= 2.0f;
        }
    }
    if( k == MAX_ITERATIONS_RESIDUAL_NRG ) {
        silk_assert( nrg == 0 );
        nrg = 1.0f;
    }

    return nrg;
}

/* Calculates residual energies of input subframes where all subframes have LPC_order   */
/* of preceding samples                                                                 */
void silk_residual_energy_FLP(
    silk_float                      nrgs[ MAX_NB_SUBFR ],               /* O    Residual energy per subframe                */
    const silk_float                x[],                                /* I    Input signal                                */
    silk_float                      a[ 2 ][ MAX_LPC_ORDER ],            /* I    AR coefs for each frame half                */
    const silk_float                gains[],                            /* I    Quantization gains                          */
    const opus_int                  subfr_length,                       /* I    Subframe length                             */
    const opus_int                  nb_subfr,                           /* I    number of subframes                         */
    const opus_int                  LPC_order                           /* I    LPC order                                   */
)
{
    opus_int     shift;
    silk_float   *LPC_res_ptr, LPC_res[ ( MAX_FRAME_LENGTH + MAX_NB_SUBFR * MAX_LPC_ORDER ) / 2 ];

    LPC_res_ptr = LPC_res + LPC_order;
    shift = LPC_order + subfr_length;

    /* Filter input to create the LPC residual for each frame half, and measure subframe energies */
    silk_LPC_analysis_filter_FLP( LPC_res, a[ 0 ], x + 0 * shift, 2 * shift, LPC_order );
    nrgs[ 0 ] = ( silk_float )( gains[ 0 ] * gains[ 0 ] * silk_energy_FLP( LPC_res_ptr + 0 * shift, subfr_length ) );
    nrgs[ 1 ] = ( silk_float )( gains[ 1 ] * gains[ 1 ] * silk_energy_FLP( LPC_res_ptr + 1 * shift, subfr_length ) );

    if( nb_subfr == MAX_NB_SUBFR ) {
        silk_LPC_analysis_filter_FLP( LPC_res, a[ 1 ], x + 2 * shift, 2 * shift, LPC_order );
        nrgs[ 2 ] = ( silk_float )( gains[ 2 ] * gains[ 2 ] * silk_energy_FLP( LPC_res_ptr + 0 * shift, subfr_length ) );
        nrgs[ 3 ] = ( silk_float )( gains[ 3 ] * gains[ 3 ] * silk_energy_FLP( LPC_res_ptr + 1 * shift, subfr_length ) );
    }
}
