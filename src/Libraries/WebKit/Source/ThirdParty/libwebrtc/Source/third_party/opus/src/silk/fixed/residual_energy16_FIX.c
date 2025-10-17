/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 27, 2022.
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

#include "main_FIX.h"

/* Residual energy: nrg = wxx - 2 * wXx * c + c' * wXX * c */
opus_int32 silk_residual_energy16_covar_FIX(
    const opus_int16                *c,                                     /* I    Prediction vector                                                           */
    const opus_int32                *wXX,                                   /* I    Correlation matrix                                                          */
    const opus_int32                *wXx,                                   /* I    Correlation vector                                                          */
    opus_int32                      wxx,                                    /* I    Signal energy                                                               */
    opus_int                        D,                                      /* I    Dimension                                                                   */
    opus_int                        cQ                                      /* I    Q value for c vector 0 - 15                                                 */
)
{
    opus_int   i, j, lshifts, Qxtra;
    opus_int32 c_max, w_max, tmp, tmp2, nrg;
    opus_int   cn[ MAX_MATRIX_SIZE ];
    const opus_int32 *pRow;

    /* Safety checks */
    celt_assert( D >=  0 );
    celt_assert( D <= 16 );
    celt_assert( cQ >  0 );
    celt_assert( cQ < 16 );

    lshifts = 16 - cQ;
    Qxtra = lshifts;

    c_max = 0;
    for( i = 0; i < D; i++ ) {
        c_max = silk_max_32( c_max, silk_abs( (opus_int32)c[ i ] ) );
    }
    Qxtra = silk_min_int( Qxtra, silk_CLZ32( c_max ) - 17 );

    w_max = silk_max_32( wXX[ 0 ], wXX[ D * D - 1 ] );
    Qxtra = silk_min_int( Qxtra, silk_CLZ32( silk_MUL( D, silk_RSHIFT( silk_SMULWB( w_max, c_max ), 4 ) ) ) - 5 );
    Qxtra = silk_max_int( Qxtra, 0 );
    for( i = 0; i < D; i++ ) {
        cn[ i ] = silk_LSHIFT( ( opus_int )c[ i ], Qxtra );
        silk_assert( silk_abs(cn[i]) <= ( silk_int16_MAX + 1 ) ); /* Check that silk_SMLAWB can be used */
    }
    lshifts -= Qxtra;

    /* Compute wxx - 2 * wXx * c */
    tmp = 0;
    for( i = 0; i < D; i++ ) {
        tmp = silk_SMLAWB( tmp, wXx[ i ], cn[ i ] );
    }
    nrg = silk_RSHIFT( wxx, 1 + lshifts ) - tmp;                         /* Q: -lshifts - 1 */

    /* Add c' * wXX * c, assuming wXX is symmetric */
    tmp2 = 0;
    for( i = 0; i < D; i++ ) {
        tmp = 0;
        pRow = &wXX[ i * D ];
        for( j = i + 1; j < D; j++ ) {
            tmp = silk_SMLAWB( tmp, pRow[ j ], cn[ j ] );
        }
        tmp  = silk_SMLAWB( tmp,  silk_RSHIFT( pRow[ i ], 1 ), cn[ i ] );
        tmp2 = silk_SMLAWB( tmp2, tmp,                        cn[ i ] );
    }
    nrg = silk_ADD_LSHIFT32( nrg, tmp2, lshifts );                       /* Q: -lshifts - 1 */

    /* Keep one bit free always, because we add them for LSF interpolation */
    if( nrg < 1 ) {
        nrg = 1;
    } else if( nrg > silk_RSHIFT( silk_int32_MAX, lshifts + 2 ) ) {
        nrg = silk_int32_MAX >> 1;
    } else {
        nrg = silk_LSHIFT( nrg, lshifts + 1 );                           /* Q0 */
    }
    return nrg;

}
