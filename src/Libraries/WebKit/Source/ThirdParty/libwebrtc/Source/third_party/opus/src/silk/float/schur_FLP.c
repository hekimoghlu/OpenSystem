/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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

#include "SigProc_FLP.h"

silk_float silk_schur_FLP(                  /* O    returns residual energy                                     */
    silk_float          refl_coef[],        /* O    reflection coefficients (length order)                      */
    const silk_float    auto_corr[],        /* I    autocorrelation sequence (length order+1)                   */
    opus_int            order               /* I    order                                                       */
)
{
    opus_int   k, n;
    double C[ SILK_MAX_ORDER_LPC + 1 ][ 2 ];
    double Ctmp1, Ctmp2, rc_tmp;

    celt_assert( order >= 0 && order <= SILK_MAX_ORDER_LPC );

    /* Copy correlations */
    k = 0;
    do {
        C[ k ][ 0 ] = C[ k ][ 1 ] = auto_corr[ k ];
    } while( ++k <= order );

    for( k = 0; k < order; k++ ) {
        /* Get reflection coefficient */
        rc_tmp = -C[ k + 1 ][ 0 ] / silk_max_float( C[ 0 ][ 1 ], 1e-9f );

        /* Save the output */
        refl_coef[ k ] = (silk_float)rc_tmp;

        /* Update correlations */
        for( n = 0; n < order - k; n++ ) {
            Ctmp1 = C[ n + k + 1 ][ 0 ];
            Ctmp2 = C[ n ][ 1 ];
            C[ n + k + 1 ][ 0 ] = Ctmp1 + Ctmp2 * rc_tmp;
            C[ n ][ 1 ]         = Ctmp2 + Ctmp1 * rc_tmp;
        }
    }

    /* Return residual energy */
    return (silk_float)C[ 0 ][ 1 ];
}
