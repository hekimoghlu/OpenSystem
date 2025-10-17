/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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

#include "main_FLP.h"

/* Apply sine window to signal vector   */
/* Window types:                        */
/*  1 -> sine window from 0 to pi/2     */
/*  2 -> sine window from pi/2 to pi    */
void silk_apply_sine_window_FLP(
    silk_float                      px_win[],                           /* O    Pointer to windowed signal                  */
    const silk_float                px[],                               /* I    Pointer to input signal                     */
    const opus_int                  win_type,                           /* I    Selects a window type                       */
    const opus_int                  length                              /* I    Window length, multiple of 4                */
)
{
    opus_int   k;
    silk_float freq, c, S0, S1;

    celt_assert( win_type == 1 || win_type == 2 );

    /* Length must be multiple of 4 */
    celt_assert( ( length & 3 ) == 0 );

    freq = PI / ( length + 1 );

    /* Approximation of 2 * cos(f) */
    c = 2.0f - freq * freq;

    /* Initialize state */
    if( win_type < 2 ) {
        /* Start from 0 */
        S0 = 0.0f;
        /* Approximation of sin(f) */
        S1 = freq;
    } else {
        /* Start from 1 */
        S0 = 1.0f;
        /* Approximation of cos(f) */
        S1 = 0.5f * c;
    }

    /* Uses the recursive equation:   sin(n*f) = 2 * cos(f) * sin((n-1)*f) - sin((n-2)*f)   */
    /* 4 samples at a time */
    for( k = 0; k < length; k += 4 ) {
        px_win[ k + 0 ] = px[ k + 0 ] * 0.5f * ( S0 + S1 );
        px_win[ k + 1 ] = px[ k + 1 ] * S1;
        S0 = c * S1 - S0;
        px_win[ k + 2 ] = px[ k + 2 ] * 0.5f * ( S1 + S0 );
        px_win[ k + 3 ] = px[ k + 3 ] * S0;
        S1 = c * S0 - S1;
    }
}
