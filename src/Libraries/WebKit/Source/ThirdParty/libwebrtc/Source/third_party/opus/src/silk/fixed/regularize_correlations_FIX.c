/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

/* Add noise to matrix diagonal */
void silk_regularize_correlations_FIX(
    opus_int32                      *XX,                                    /* I/O  Correlation matrices                                                        */
    opus_int32                      *xx,                                    /* I/O  Correlation values                                                          */
    opus_int32                      noise,                                  /* I    Noise to add                                                                */
    opus_int                        D                                       /* I    Dimension of XX                                                             */
)
{
    opus_int i;
    for( i = 0; i < D; i++ ) {
        matrix_ptr( &XX[ 0 ], i, i, D ) = silk_ADD32( matrix_ptr( &XX[ 0 ], i, i, D ), noise );
    }
    xx[ 0 ] += noise;
}
