/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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

/* Insertion sort (fast for already almost sorted arrays):  */
/* Best case:  O(n)   for an already sorted array           */
/* Worst case: O(n^2) for an inversely sorted array         */

#include "typedef.h"
#include "SigProc_FLP.h"

void silk_insertion_sort_decreasing_FLP(
    silk_float          *a,                 /* I/O  Unsorted / Sorted vector                                    */
    opus_int            *idx,               /* O    Index vector for the sorted elements                        */
    const opus_int      L,                  /* I    Vector length                                               */
    const opus_int      K                   /* I    Number of correctly sorted positions                        */
)
{
    silk_float value;
    opus_int   i, j;

    /* Safety checks */
    celt_assert( K >  0 );
    celt_assert( L >  0 );
    celt_assert( L >= K );

    /* Write start indices in index vector */
    for( i = 0; i < K; i++ ) {
        idx[ i ] = i;
    }

    /* Sort vector elements by value, decreasing order */
    for( i = 1; i < K; i++ ) {
        value = a[ i ];
        for( j = i - 1; ( j >= 0 ) && ( value > a[ j ] ); j-- ) {
            a[ j + 1 ]   = a[ j ];      /* Shift value */
            idx[ j + 1 ] = idx[ j ];    /* Shift index */
        }
        a[ j + 1 ]   = value;   /* Write value */
        idx[ j + 1 ] = i;       /* Write index */
    }

    /* If less than L values are asked check the remaining values,      */
    /* but only spend CPU to ensure that the K first values are correct */
    for( i = K; i < L; i++ ) {
        value = a[ i ];
        if( value > a[ K - 1 ] ) {
            for( j = K - 2; ( j >= 0 ) && ( value > a[ j ] ); j-- ) {
                a[ j + 1 ]   = a[ j ];      /* Shift value */
                idx[ j + 1 ] = idx[ j ];    /* Shift index */
            }
            a[ j + 1 ]   = value;   /* Write value */
            idx[ j + 1 ] = i;       /* Write index */
        }
    }
}
