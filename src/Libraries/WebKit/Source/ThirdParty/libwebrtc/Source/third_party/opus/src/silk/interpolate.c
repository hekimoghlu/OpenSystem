/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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

/* Interpolate two vectors */
void silk_interpolate(
    opus_int16                  xi[ MAX_LPC_ORDER ],            /* O    interpolated vector                         */
    const opus_int16            x0[ MAX_LPC_ORDER ],            /* I    first vector                                */
    const opus_int16            x1[ MAX_LPC_ORDER ],            /* I    second vector                               */
    const opus_int              ifact_Q2,                       /* I    interp. factor, weight on 2nd vector        */
    const opus_int              d                               /* I    number of parameters                        */
)
{
    opus_int i;

    celt_assert( ifact_Q2 >= 0 );
    celt_assert( ifact_Q2 <= 4 );

    for( i = 0; i < d; i++ ) {
        xi[ i ] = (opus_int16)silk_ADD_RSHIFT( x0[ i ], silk_SMULBB( x1[ i ] - x0[ i ], ifact_Q2 ), 2 );
    }
}
