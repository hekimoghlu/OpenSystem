/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 10, 2023.
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
#ifndef SILK_FIX_RESAMPLER_ROM_H
#define SILK_FIX_RESAMPLER_ROM_H

#ifdef  __cplusplus
extern "C"
{
#endif

#include "typedef.h"
#include "resampler_structs.h"

#define RESAMPLER_DOWN_ORDER_FIR0               18
#define RESAMPLER_DOWN_ORDER_FIR1               24
#define RESAMPLER_DOWN_ORDER_FIR2               36
#define RESAMPLER_ORDER_FIR_12                  8

/* Tables for 2x downsampler */
static const opus_int16 silk_resampler_down2_0 = 9872;
static const opus_int16 silk_resampler_down2_1 = 39809 - 65536;

/* Tables for 2x upsampler, high quality */
static const opus_int16 silk_resampler_up2_hq_0[ 3 ] = { 1746, 14986, 39083 - 65536 };
static const opus_int16 silk_resampler_up2_hq_1[ 3 ] = { 6854, 25769, 55542 - 65536 };

/* Tables with IIR and FIR coefficients for fractional downsamplers */
extern const opus_int16 silk_Resampler_3_4_COEFS[ 2 + 3 * RESAMPLER_DOWN_ORDER_FIR0 / 2 ];
extern const opus_int16 silk_Resampler_2_3_COEFS[ 2 + 2 * RESAMPLER_DOWN_ORDER_FIR0 / 2 ];
extern const opus_int16 silk_Resampler_1_2_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR1 / 2 ];
extern const opus_int16 silk_Resampler_1_3_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR2 / 2 ];
extern const opus_int16 silk_Resampler_1_4_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR2 / 2 ];
extern const opus_int16 silk_Resampler_1_6_COEFS[ 2 +     RESAMPLER_DOWN_ORDER_FIR2 / 2 ];
extern const opus_int16 silk_Resampler_2_3_COEFS_LQ[ 2 + 2 * 2 ];

/* Table with interplation fractions of 1/24, 3/24, ..., 23/24 */
extern const opus_int16 silk_resampler_frac_FIR_12[ 12 ][ RESAMPLER_ORDER_FIR_12 / 2 ];

#ifdef  __cplusplus
}
#endif

#endif /* SILK_FIX_RESAMPLER_ROM_H */
