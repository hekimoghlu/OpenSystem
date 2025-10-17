/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#include "tuning_parameters.h"
#include "define.h"

#define MAX_FRAME_SIZE              384 /* subfr_length * nb_subfr = ( 0.005 * 16000 + 16 ) * 4 = 384*/

/* Compute reflection coefficients from input signal */
silk_float silk_burg_modified_FLP(          /* O    returns residual energy                                     */
    silk_float          A[],                /* O    prediction coefficients (length order)                      */
    const silk_float    x[],                /* I    input signal, length: nb_subfr*(D+L_sub)                    */
    const silk_float    minInvGain,         /* I    minimum inverse prediction gain                             */
    const opus_int      subfr_length,       /* I    input signal subframe length (incl. D preceding samples)    */
    const opus_int      nb_subfr,           /* I    number of subframes stacked in x                            */
    const opus_int      D                   /* I    order                                                       */
)
{
    opus_int         k, n, s, reached_max_gain;
    double           C0, invGain, num, nrg_f, nrg_b, rc, Atmp, tmp1, tmp2;
    const silk_float *x_ptr;
    double           C_first_row[ SILK_MAX_ORDER_LPC ], C_last_row[ SILK_MAX_ORDER_LPC ];
    double           CAf[ SILK_MAX_ORDER_LPC + 1 ], CAb[ SILK_MAX_ORDER_LPC + 1 ];
    double           Af[ SILK_MAX_ORDER_LPC ];

    celt_assert( subfr_length * nb_subfr <= MAX_FRAME_SIZE );

    /* Compute autocorrelations, added over subframes */
    C0 = silk_energy_FLP( x, nb_subfr * subfr_length );
    silk_memset( C_first_row, 0, SILK_MAX_ORDER_LPC * sizeof( double ) );
    for( s = 0; s < nb_subfr; s++ ) {
        x_ptr = x + s * subfr_length;
        for( n = 1; n < D + 1; n++ ) {
            C_first_row[ n - 1 ] += silk_inner_product_FLP( x_ptr, x_ptr + n, subfr_length - n );
        }
    }
    silk_memcpy( C_last_row, C_first_row, SILK_MAX_ORDER_LPC * sizeof( double ) );

    /* Initialize */
    CAb[ 0 ] = CAf[ 0 ] = C0 + FIND_LPC_COND_FAC * C0 + 1e-9f;
    invGain = 1.0f;
    reached_max_gain = 0;
    for( n = 0; n < D; n++ ) {
        /* Update first row of correlation matrix (without first element) */
        /* Update last row of correlation matrix (without last element, stored in reversed order) */
        /* Update C * Af */
        /* Update C * flipud(Af) (stored in reversed order) */
        for( s = 0; s < nb_subfr; s++ ) {
            x_ptr = x + s * subfr_length;
            tmp1 = x_ptr[ n ];
            tmp2 = x_ptr[ subfr_length - n - 1 ];
            for( k = 0; k < n; k++ ) {
                C_first_row[ k ] -= x_ptr[ n ] * x_ptr[ n - k - 1 ];
                C_last_row[ k ]  -= x_ptr[ subfr_length - n - 1 ] * x_ptr[ subfr_length - n + k ];
                Atmp = Af[ k ];
                tmp1 += x_ptr[ n - k - 1 ] * Atmp;
                tmp2 += x_ptr[ subfr_length - n + k ] * Atmp;
            }
            for( k = 0; k <= n; k++ ) {
                CAf[ k ] -= tmp1 * x_ptr[ n - k ];
                CAb[ k ] -= tmp2 * x_ptr[ subfr_length - n + k - 1 ];
            }
        }
        tmp1 = C_first_row[ n ];
        tmp2 = C_last_row[ n ];
        for( k = 0; k < n; k++ ) {
            Atmp = Af[ k ];
            tmp1 += C_last_row[  n - k - 1 ] * Atmp;
            tmp2 += C_first_row[ n - k - 1 ] * Atmp;
        }
        CAf[ n + 1 ] = tmp1;
        CAb[ n + 1 ] = tmp2;

        /* Calculate nominator and denominator for the next order reflection (parcor) coefficient */
        num = CAb[ n + 1 ];
        nrg_b = CAb[ 0 ];
        nrg_f = CAf[ 0 ];
        for( k = 0; k < n; k++ ) {
            Atmp = Af[ k ];
            num   += CAb[ n - k ] * Atmp;
            nrg_b += CAb[ k + 1 ] * Atmp;
            nrg_f += CAf[ k + 1 ] * Atmp;
        }
        silk_assert( nrg_f > 0.0 );
        silk_assert( nrg_b > 0.0 );

        /* Calculate the next order reflection (parcor) coefficient */
        rc = -2.0 * num / ( nrg_f + nrg_b );
        silk_assert( rc > -1.0 && rc < 1.0 );

        /* Update inverse prediction gain */
        tmp1 = invGain * ( 1.0 - rc * rc );
        if( tmp1 <= minInvGain ) {
            /* Max prediction gain exceeded; set reflection coefficient such that max prediction gain is exactly hit */
            rc = sqrt( 1.0 - minInvGain / invGain );
            if( num > 0 ) {
                /* Ensure adjusted reflection coefficients has the original sign */
                rc = -rc;
            }
            invGain = minInvGain;
            reached_max_gain = 1;
        } else {
            invGain = tmp1;
        }

        /* Update the AR coefficients */
        for( k = 0; k < (n + 1) >> 1; k++ ) {
            tmp1 = Af[ k ];
            tmp2 = Af[ n - k - 1 ];
            Af[ k ]         = tmp1 + rc * tmp2;
            Af[ n - k - 1 ] = tmp2 + rc * tmp1;
        }
        Af[ n ] = rc;

        if( reached_max_gain ) {
            /* Reached max prediction gain; set remaining coefficients to zero and exit loop */
            for( k = n + 1; k < D; k++ ) {
                Af[ k ] = 0.0;
            }
            break;
        }

        /* Update C * Af and C * Ab */
        for( k = 0; k <= n + 1; k++ ) {
            tmp1 = CAf[ k ];
            CAf[ k ]          += rc * CAb[ n - k + 1 ];
            CAb[ n - k + 1  ] += rc * tmp1;
        }
    }

    if( reached_max_gain ) {
        /* Convert to silk_float */
        for( k = 0; k < D; k++ ) {
            A[ k ] = (silk_float)( -Af[ k ] );
        }
        /* Subtract energy of preceding samples from C0 */
        for( s = 0; s < nb_subfr; s++ ) {
            C0 -= silk_energy_FLP( x + s * subfr_length, D );
        }
        /* Approximate residual energy */
        nrg_f = C0 * invGain;
    } else {
        /* Compute residual energy and store coefficients as silk_float */
        nrg_f = CAf[ 0 ];
        tmp1 = 1.0;
        for( k = 0; k < D; k++ ) {
            Atmp = Af[ k ];
            nrg_f += CAf[ k + 1 ] * Atmp;
            tmp1  += Atmp * Atmp;
            A[ k ] = (silk_float)(-Atmp);
        }
        nrg_f -= FIND_LPC_COND_FAC * C0 * tmp1;
    }

    /* Return residual energy */
    return (silk_float)nrg_f;
}
