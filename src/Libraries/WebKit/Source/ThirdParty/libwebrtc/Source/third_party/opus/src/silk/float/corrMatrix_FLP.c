/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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

/**********************************************************************
 * Correlation matrix computations for LS estimate.
 **********************************************************************/

#include "main_FLP.h"

/* Calculates correlation vector X'*t */
void silk_corrVector_FLP(
    const silk_float                *x,                                 /* I    x vector [L+order-1] used to create X       */
    const silk_float                *t,                                 /* I    Target vector [L]                           */
    const opus_int                  L,                                  /* I    Length of vecors                            */
    const opus_int                  Order,                              /* I    Max lag for correlation                     */
    silk_float                      *Xt                                 /* O    X'*t correlation vector [order]             */
)
{
    opus_int lag;
    const silk_float *ptr1;

    ptr1 = &x[ Order - 1 ];                     /* Points to first sample of column 0 of X: X[:,0] */
    for( lag = 0; lag < Order; lag++ ) {
        /* Calculate X[:,lag]'*t */
        Xt[ lag ] = (silk_float)silk_inner_product_FLP( ptr1, t, L );
        ptr1--;                                 /* Next column of X */
    }
}

/* Calculates correlation matrix X'*X */
void silk_corrMatrix_FLP(
    const silk_float                *x,                                 /* I    x vector [ L+order-1 ] used to create X     */
    const opus_int                  L,                                  /* I    Length of vectors                           */
    const opus_int                  Order,                              /* I    Max lag for correlation                     */
    silk_float                      *XX                                 /* O    X'*X correlation matrix [order x order]     */
)
{
    opus_int j, lag;
    double  energy;
    const silk_float *ptr1, *ptr2;

    ptr1 = &x[ Order - 1 ];                     /* First sample of column 0 of X */
    energy = silk_energy_FLP( ptr1, L );  /* X[:,0]'*X[:,0] */
    matrix_ptr( XX, 0, 0, Order ) = ( silk_float )energy;
    for( j = 1; j < Order; j++ ) {
        /* Calculate X[:,j]'*X[:,j] */
        energy += ptr1[ -j ] * ptr1[ -j ] - ptr1[ L - j ] * ptr1[ L - j ];
        matrix_ptr( XX, j, j, Order ) = ( silk_float )energy;
    }

    ptr2 = &x[ Order - 2 ];                     /* First sample of column 1 of X */
    for( lag = 1; lag < Order; lag++ ) {
        /* Calculate X[:,0]'*X[:,lag] */
        energy = silk_inner_product_FLP( ptr1, ptr2, L );
        matrix_ptr( XX, lag, 0, Order ) = ( silk_float )energy;
        matrix_ptr( XX, 0, lag, Order ) = ( silk_float )energy;
        /* Calculate X[:,j]'*X[:,j + lag] */
        for( j = 1; j < ( Order - lag ); j++ ) {
            energy += ptr1[ -j ] * ptr2[ -j ] - ptr1[ L - j ] * ptr2[ L - j ];
            matrix_ptr( XX, lag + j, j, Order ) = ( silk_float )energy;
            matrix_ptr( XX, j, lag + j, Order ) = ( silk_float )energy;
        }
        ptr2--;                                 /* Next column of X */
    }
}
