/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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

/* copy and multiply a vector by a constant */
void silk_scale_copy_vector_FLP(
    silk_float          *data_out,
    const silk_float    *data_in,
    silk_float          gain,
    opus_int            dataSize
)
{
    opus_int  i, dataSize4;

    /* 4x unrolled loop */
    dataSize4 = dataSize & 0xFFFC;
    for( i = 0; i < dataSize4; i += 4 ) {
        data_out[ i + 0 ] = gain * data_in[ i + 0 ];
        data_out[ i + 1 ] = gain * data_in[ i + 1 ];
        data_out[ i + 2 ] = gain * data_in[ i + 2 ];
        data_out[ i + 3 ] = gain * data_in[ i + 3 ];
    }

    /* any remaining elements */
    for( ; i < dataSize; i++ ) {
        data_out[ i ] = gain * data_in[ i ];
    }
}
