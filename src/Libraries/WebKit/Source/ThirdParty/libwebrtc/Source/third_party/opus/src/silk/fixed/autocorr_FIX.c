/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#include "celt_lpc.h"

/* Compute autocorrelation */
void silk_autocorr(
    opus_int32                  *results,           /* O    Result (length correlationCount)                            */
    opus_int                    *scale,             /* O    Scaling of the correlation vector                           */
    const opus_int16            *inputData,         /* I    Input data to correlate                                     */
    const opus_int              inputDataSize,      /* I    Length of input                                             */
    const opus_int              correlationCount,   /* I    Number of correlation taps to compute                       */
    int                         arch                /* I    Run-time architecture                                       */
)
{
    opus_int   corrCount;
    corrCount = silk_min_int( inputDataSize, correlationCount );
    *scale = _celt_autocorr(inputData, results, NULL, 0, corrCount-1, inputDataSize, arch);
}
