/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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
/* Conversion between prediction filter coefficients and NLSFs  */
/* Requires the order to be an even number                      */
/* A piecewise linear approximation maps LSF <-> cos(LSF)       */
/* Therefore the result is not accurate NLSFs, but the two      */
/* functions are accurate inverses of each other                */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "SigProc_FIX.h"
#include "tables.h"

/* Number of binary divisions, when not in low complexity mode */
#define BIN_DIV_STEPS_A2NLSF_FIX      3 /* must be no higher than 16 - log2( LSF_COS_TAB_SZ_FIX ) */
#define MAX_ITERATIONS_A2NLSF_FIX    16

/* Helper function for A2NLSF(..)                    */
/* Transforms polynomials from cos(n*f) to cos(f)^n  */
static OPUS_INLINE void silk_A2NLSF_trans_poly(
    opus_int32          *p,                     /* I/O    Polynomial                                */
    const opus_int      dd                      /* I      Polynomial order (= filter order / 2 )    */
)
{
    opus_int k, n;

    for( k = 2; k <= dd; k++ ) {
        for( n = dd; n > k; n-- ) {
            p[ n - 2 ] -= p[ n ];
        }
        p[ k - 2 ] -= silk_LSHIFT( p[ k ], 1 );
    }
}
/* Helper function for A2NLSF(..) */
/* Polynomial evaluation          */
static OPUS_INLINE opus_int32 silk_A2NLSF_eval_poly( /* return the polynomial evaluation, in Q16     */
    opus_int32          *p,                     /* I    Polynomial, Q16                         */
    const opus_int32    x,                      /* I    Evaluation point, Q12                   */
    const opus_int      dd                      /* I    Order                                   */
)
{
    opus_int   n;
    opus_int32 x_Q16, y32;

    y32 = p[ dd ];                                  /* Q16 */
    x_Q16 = silk_LSHIFT( x, 4 );

    if ( opus_likely( 8 == dd ) )
    {
        y32 = silk_SMLAWW( p[ 7 ], y32, x_Q16 );
        y32 = silk_SMLAWW( p[ 6 ], y32, x_Q16 );
        y32 = silk_SMLAWW( p[ 5 ], y32, x_Q16 );
        y32 = silk_SMLAWW( p[ 4 ], y32, x_Q16 );
        y32 = silk_SMLAWW( p[ 3 ], y32, x_Q16 );
        y32 = silk_SMLAWW( p[ 2 ], y32, x_Q16 );
        y32 = silk_SMLAWW( p[ 1 ], y32, x_Q16 );
        y32 = silk_SMLAWW( p[ 0 ], y32, x_Q16 );
    }
    else
    {
        for( n = dd - 1; n >= 0; n-- ) {
            y32 = silk_SMLAWW( p[ n ], y32, x_Q16 );    /* Q16 */
        }
    }
    return y32;
}

static OPUS_INLINE void silk_A2NLSF_init(
     const opus_int32    *a_Q16,
     opus_int32          *P,
     opus_int32          *Q,
     const opus_int      dd
)
{
    opus_int k;

    /* Convert filter coefs to even and odd polynomials */
    P[dd] = silk_LSHIFT( 1, 16 );
    Q[dd] = silk_LSHIFT( 1, 16 );
    for( k = 0; k < dd; k++ ) {
        P[ k ] = -a_Q16[ dd - k - 1 ] - a_Q16[ dd + k ];    /* Q16 */
        Q[ k ] = -a_Q16[ dd - k - 1 ] + a_Q16[ dd + k ];    /* Q16 */
    }

    /* Divide out zeros as we have that for even filter orders, */
    /* z =  1 is always a root in Q, and                        */
    /* z = -1 is always a root in P                             */
    for( k = dd; k > 0; k-- ) {
        P[ k - 1 ] -= P[ k ];
        Q[ k - 1 ] += Q[ k ];
    }

    /* Transform polynomials from cos(n*f) to cos(f)^n */
    silk_A2NLSF_trans_poly( P, dd );
    silk_A2NLSF_trans_poly( Q, dd );
}

/* Compute Normalized Line Spectral Frequencies (NLSFs) from whitening filter coefficients      */
/* If not all roots are found, the a_Q16 coefficients are bandwidth expanded until convergence. */
void silk_A2NLSF(
    opus_int16                  *NLSF,              /* O    Normalized Line Spectral Frequencies in Q15 (0..2^15-1) [d] */
    opus_int32                  *a_Q16,             /* I/O  Monic whitening filter coefficients in Q16 [d]              */
    const opus_int              d                   /* I    Filter order (must be even)                                 */
)
{
    opus_int   i, k, m, dd, root_ix, ffrac;
    opus_int32 xlo, xhi, xmid;
    opus_int32 ylo, yhi, ymid, thr;
    opus_int32 nom, den;
    opus_int32 P[ SILK_MAX_ORDER_LPC / 2 + 1 ];
    opus_int32 Q[ SILK_MAX_ORDER_LPC / 2 + 1 ];
    opus_int32 *PQ[ 2 ];
    opus_int32 *p;

    /* Store pointers to array */
    PQ[ 0 ] = P;
    PQ[ 1 ] = Q;

    dd = silk_RSHIFT( d, 1 );

    silk_A2NLSF_init( a_Q16, P, Q, dd );

    /* Find roots, alternating between P and Q */
    p = P;                          /* Pointer to polynomial */

    xlo = silk_LSFCosTab_FIX_Q12[ 0 ]; /* Q12*/
    ylo = silk_A2NLSF_eval_poly( p, xlo, dd );

    if( ylo < 0 ) {
        /* Set the first NLSF to zero and move on to the next */
        NLSF[ 0 ] = 0;
        p = Q;                      /* Pointer to polynomial */
        ylo = silk_A2NLSF_eval_poly( p, xlo, dd );
        root_ix = 1;                /* Index of current root */
    } else {
        root_ix = 0;                /* Index of current root */
    }
    k = 1;                          /* Loop counter */
    i = 0;                          /* Counter for bandwidth expansions applied */
    thr = 0;
    while( 1 ) {
        /* Evaluate polynomial */
        xhi = silk_LSFCosTab_FIX_Q12[ k ]; /* Q12 */
        yhi = silk_A2NLSF_eval_poly( p, xhi, dd );

        /* Detect zero crossing */
        if( ( ylo <= 0 && yhi >= thr ) || ( ylo >= 0 && yhi <= -thr ) ) {
            if( yhi == 0 ) {
                /* If the root lies exactly at the end of the current       */
                /* interval, look for the next root in the next interval    */
                thr = 1;
            } else {
                thr = 0;
            }
            /* Binary division */
            ffrac = -256;
            for( m = 0; m < BIN_DIV_STEPS_A2NLSF_FIX; m++ ) {
                /* Evaluate polynomial */
                xmid = silk_RSHIFT_ROUND( xlo + xhi, 1 );
                ymid = silk_A2NLSF_eval_poly( p, xmid, dd );

                /* Detect zero crossing */
                if( ( ylo <= 0 && ymid >= 0 ) || ( ylo >= 0 && ymid <= 0 ) ) {
                    /* Reduce frequency */
                    xhi = xmid;
                    yhi = ymid;
                } else {
                    /* Increase frequency */
                    xlo = xmid;
                    ylo = ymid;
                    ffrac = silk_ADD_RSHIFT( ffrac, 128, m );
                }
            }

            /* Interpolate */
            if( silk_abs( ylo ) < 65536 ) {
                /* Avoid dividing by zero */
                den = ylo - yhi;
                nom = silk_LSHIFT( ylo, 8 - BIN_DIV_STEPS_A2NLSF_FIX ) + silk_RSHIFT( den, 1 );
                if( den != 0 ) {
                    ffrac += silk_DIV32( nom, den );
                }
            } else {
                /* No risk of dividing by zero because abs(ylo - yhi) >= abs(ylo) >= 65536 */
                ffrac += silk_DIV32( ylo, silk_RSHIFT( ylo - yhi, 8 - BIN_DIV_STEPS_A2NLSF_FIX ) );
            }
            NLSF[ root_ix ] = (opus_int16)silk_min_32( silk_LSHIFT( (opus_int32)k, 8 ) + ffrac, silk_int16_MAX );

            silk_assert( NLSF[ root_ix ] >= 0 );

            root_ix++;        /* Next root */
            if( root_ix >= d ) {
                /* Found all roots */
                break;
            }
            /* Alternate pointer to polynomial */
            p = PQ[ root_ix & 1 ];

            /* Evaluate polynomial */
            xlo = silk_LSFCosTab_FIX_Q12[ k - 1 ]; /* Q12*/
            ylo = silk_LSHIFT( 1 - ( root_ix & 2 ), 12 );
        } else {
            /* Increment loop counter */
            k++;
            xlo = xhi;
            ylo = yhi;
            thr = 0;

            if( k > LSF_COS_TAB_SZ_FIX ) {
                i++;
                if( i > MAX_ITERATIONS_A2NLSF_FIX ) {
                    /* Set NLSFs to white spectrum and exit */
                    NLSF[ 0 ] = (opus_int16)silk_DIV32_16( 1 << 15, d + 1 );
                    for( k = 1; k < d; k++ ) {
                        NLSF[ k ] = (opus_int16)silk_ADD16( NLSF[ k-1 ], NLSF[ 0 ] );
                    }
                    return;
                }

                /* Error: Apply progressively more bandwidth expansion and run again */
                silk_bwexpander_32( a_Q16, d, 65536 - silk_LSHIFT( 1, i ) );

                silk_A2NLSF_init( a_Q16, P, Q, dd );
                p = P;                            /* Pointer to polynomial */
                xlo = silk_LSFCosTab_FIX_Q12[ 0 ]; /* Q12*/
                ylo = silk_A2NLSF_eval_poly( p, xlo, dd );
                if( ylo < 0 ) {
                    /* Set the first NLSF to zero and move on to the next */
                    NLSF[ 0 ] = 0;
                    p = Q;                        /* Pointer to polynomial */
                    ylo = silk_A2NLSF_eval_poly( p, xlo, dd );
                    root_ix = 1;                  /* Index of current root */
                } else {
                    root_ix = 0;                  /* Index of current root */
                }
                k = 1;                            /* Reset loop counter */
            }
        }
    }
}
