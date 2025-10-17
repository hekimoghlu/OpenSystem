/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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

#include "tables.h"

#ifdef __cplusplus
extern "C"
{
#endif

const opus_uint8 silk_gain_iCDF[ 3 ][ N_LEVELS_QGAIN / 8 ] =
{
{
       224,    112,     44,     15,      3,      2,      1,      0
},
{
       254,    237,    192,    132,     70,     23,      4,      0
},
{
       255,    252,    226,    155,     61,     11,      2,      0
}
};

const opus_uint8 silk_delta_gain_iCDF[ MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT + 1 ] = {
       250,    245,    234,    203,     71,     50,     42,     38,
        35,     33,     31,     29,     28,     27,     26,     25,
        24,     23,     22,     21,     20,     19,     18,     17,
        16,     15,     14,     13,     12,     11,     10,      9,
         8,      7,      6,      5,      4,      3,      2,      1,
         0
};

#ifdef __cplusplus
}
#endif
