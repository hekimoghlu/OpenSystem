/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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

/* sum of squares of a silk_float array, with result as double */
double silk_energy_FLP(
    const silk_float    *data,
    opus_int            dataSize
)
{
    opus_int i;
    double   result;

    /* 4x unrolled loop */
    result = 0.0;
    for( i = 0; i < dataSize - 3; i += 4 ) {
        result += data[ i + 0 ] * (double)data[ i + 0 ] +
                  data[ i + 1 ] * (double)data[ i + 1 ] +
                  data[ i + 2 ] * (double)data[ i + 2 ] +
                  data[ i + 3 ] * (double)data[ i + 3 ];
    }

    /* add any remaining products */
    for( ; i < dataSize; i++ ) {
        result += data[ i ] * (double)data[ i ];
    }

    silk_assert( result >= 0.0 );
    return result;
}
