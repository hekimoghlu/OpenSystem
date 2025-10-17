/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#include "SigProc_FLP.h"
#include "define.h"

/* compute inverse of LPC prediction gain, and                          */
/* test if LPC coefficients are stable (all poles within unit circle)   */
/* this code is based on silk_a2k_FLP()                                 */
silk_float silk_LPC_inverse_pred_gain_FLP(  /* O    return inverse prediction gain, energy domain               */
    const silk_float    *A,                 /* I    prediction coefficients [order]                             */
    opus_int32          order               /* I    prediction order                                            */
)
{
    opus_int   k, n;
    double     invGain, rc, rc_mult1, rc_mult2, tmp1, tmp2;
    silk_float Atmp[ SILK_MAX_ORDER_LPC ];

    silk_memcpy( Atmp, A, order * sizeof(silk_float) );

    invGain = 1.0;
    for( k = order - 1; k > 0; k-- ) {
        rc = -Atmp[ k ];
        rc_mult1 = 1.0f - rc * rc;
        invGain *= rc_mult1;
        if( invGain * MAX_PREDICTION_POWER_GAIN < 1.0f ) {
            return 0.0f;
        }
        rc_mult2 = 1.0f / rc_mult1;
        for( n = 0; n < (k + 1) >> 1; n++ ) {
            tmp1 = Atmp[ n ];
            tmp2 = Atmp[ k - n - 1 ];
            Atmp[ n ]         = (silk_float)( ( tmp1 - tmp2 * rc ) * rc_mult2 );
            Atmp[ k - n - 1 ] = (silk_float)( ( tmp2 - tmp1 * rc ) * rc_mult2 );
        }
    }
    rc = -Atmp[ 0 ];
    rc_mult1 = 1.0f - rc * rc;
    invGain *= rc_mult1;
    if( invGain * MAX_PREDICTION_POWER_GAIN < 1.0f ) {
        return 0.0f;
    }
    return (silk_float)invGain;
}
