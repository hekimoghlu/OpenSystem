/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 11, 2022.
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

#include "typedef.h"
#include "pitch_est_defines.h"

const opus_int8 silk_CB_lags_stage2_10_ms[ PE_MAX_NB_SUBFR >> 1][ PE_NB_CBKS_STAGE2_10MS ] =
{
    {0, 1, 0},
    {0, 0, 1}
};

const opus_int8 silk_CB_lags_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ PE_NB_CBKS_STAGE3_10MS ] =
{
    { 0, 0, 1,-1, 1,-1, 2,-2, 2,-2, 3,-3},
    { 0, 1, 0, 1,-1, 2,-1, 2,-2, 3,-2, 3}
};

const opus_int8 silk_Lag_range_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ 2 ] =
{
    {-3, 7},
    {-2, 7}
};

const opus_int8 silk_CB_lags_stage2[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE2_EXT ] =
{
    {0, 2,-1,-1,-1, 0, 0, 1, 1, 0, 1},
    {0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0},
    {0,-1, 2, 1, 0, 1, 1, 0, 0,-1,-1}
};

const opus_int8 silk_CB_lags_stage3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ] =
{
    {0, 0, 1,-1, 0, 1,-1, 0,-1, 1,-2, 2,-2,-2, 2,-3, 2, 3,-3,-4, 3,-4, 4, 4,-5, 5,-6,-5, 6,-7, 6, 5, 8,-9},
    {0, 0, 1, 0, 0, 0, 0, 0, 0, 0,-1, 1, 0, 0, 1,-1, 0, 1,-1,-1, 1,-1, 2, 1,-1, 2,-2,-2, 2,-2, 2, 2, 3,-3},
    {0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,-1, 1, 0, 0, 2, 1,-1, 2,-1,-1, 2,-1, 2, 2,-1, 3,-2,-2,-2, 3},
    {0, 1, 0, 0, 1, 0, 1,-1, 2,-1, 2,-1, 2, 3,-2, 3,-2,-2, 4, 4,-3, 5,-3,-4, 6,-4, 6, 5,-5, 8,-6,-5,-7, 9}
};

const opus_int8 silk_Lag_range_stage3[ SILK_PE_MAX_COMPLEX + 1 ] [ PE_MAX_NB_SUBFR ][ 2 ] =
{
    /* Lags to search for low number of stage3 cbks */
    {
        {-5,8},
        {-1,6},
        {-1,6},
        {-4,10}
    },
    /* Lags to search for middle number of stage3 cbks */
    {
        {-6,10},
        {-2,6},
        {-1,6},
        {-5,10}
    },
    /* Lags to search for max number of stage3 cbks */
    {
        {-9,12},
        {-3,7},
        {-2,7},
        {-7,13}
    }
};

const opus_int8 silk_nb_cbk_searchs_stage3[ SILK_PE_MAX_COMPLEX + 1 ] =
{
    PE_NB_CBKS_STAGE3_MIN,
    PE_NB_CBKS_STAGE3_MID,
    PE_NB_CBKS_STAGE3_MAX
};
