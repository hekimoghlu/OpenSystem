/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
#include "stack_alloc.h"

/* Calculates residual energies of input subframes where all subframes have LPC_order   */
/* of preceding samples                                                                 */
void silk_residual_energy_FIX(
          opus_int32                nrgs[ MAX_NB_SUBFR ],                   /* O    Residual energy per subframe                                                */
          opus_int                  nrgsQ[ MAX_NB_SUBFR ],                  /* O    Q value per subframe                                                        */
    const opus_int16                x[],                                    /* I    Input signal                                                                */
          opus_int16                a_Q12[ 2 ][ MAX_LPC_ORDER ],            /* I    AR coefs for each frame half                                                */
    const opus_int32                gains[ MAX_NB_SUBFR ],                  /* I    Quantization gains                                                          */
    const opus_int                  subfr_length,                           /* I    Subframe length                                                             */
    const opus_int                  nb_subfr,                               /* I    Number of subframes                                                         */
    const opus_int                  LPC_order,                              /* I    LPC order                                                                   */
          int                       arch                                    /* I    Run-time architecture                                                       */
)
{
    opus_int         offset, i, j, rshift, lz1, lz2;
    opus_int16       *LPC_res_ptr;
    VARDECL( opus_int16, LPC_res );
    const opus_int16 *x_ptr;
    opus_int32       tmp32;
    SAVE_STACK;

    x_ptr  = x;
    offset = LPC_order + subfr_length;

    /* Filter input to create the LPC residual for each frame half, and measure subframe energies */
    ALLOC( LPC_res, ( MAX_NB_SUBFR >> 1 ) * offset, opus_int16 );
    celt_assert( ( nb_subfr >> 1 ) * ( MAX_NB_SUBFR >> 1 ) == nb_subfr );
    for( i = 0; i < nb_subfr >> 1; i++ ) {
        /* Calculate half frame LPC residual signal including preceding samples */
        silk_LPC_analysis_filter( LPC_res, x_ptr, a_Q12[ i ], ( MAX_NB_SUBFR >> 1 ) * offset, LPC_order, arch );

        /* Point to first subframe of the just calculated LPC residual signal */
        LPC_res_ptr = LPC_res + LPC_order;
        for( j = 0; j < ( MAX_NB_SUBFR >> 1 ); j++ ) {
            /* Measure subframe energy */
            silk_sum_sqr_shift( &nrgs[ i * ( MAX_NB_SUBFR >> 1 ) + j ], &rshift, LPC_res_ptr, subfr_length );

            /* Set Q values for the measured energy */
            nrgsQ[ i * ( MAX_NB_SUBFR >> 1 ) + j ] = -rshift;

            /* Move to next subframe */
            LPC_res_ptr += offset;
        }
        /* Move to next frame half */
        x_ptr += ( MAX_NB_SUBFR >> 1 ) * offset;
    }

    /* Apply the squared subframe gains */
    for( i = 0; i < nb_subfr; i++ ) {
        /* Fully upscale gains and energies */
        lz1 = silk_CLZ32( nrgs[  i ] ) - 1;
        lz2 = silk_CLZ32( gains[ i ] ) - 1;

        tmp32 = silk_LSHIFT32( gains[ i ], lz2 );

        /* Find squared gains */
        tmp32 = silk_SMMUL( tmp32, tmp32 ); /* Q( 2 * lz2 - 32 )*/

        /* Scale energies */
        nrgs[ i ] = silk_SMMUL( tmp32, silk_LSHIFT32( nrgs[ i ], lz1 ) ); /* Q( nrgsQ[ i ] + lz1 + 2 * lz2 - 32 - 32 )*/
        nrgsQ[ i ] += lz1 + 2 * lz2 - 32 - 32;
    }
    RESTORE_STACK;
}
