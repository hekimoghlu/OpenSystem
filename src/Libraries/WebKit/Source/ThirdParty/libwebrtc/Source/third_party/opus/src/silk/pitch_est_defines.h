/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#ifndef SILK_PE_DEFINES_H
#define SILK_PE_DEFINES_H

#include "SigProc_FIX.h"

/********************************************************/
/* Definitions for pitch estimator                      */
/********************************************************/

#define PE_MAX_FS_KHZ               16 /* Maximum sampling frequency used */

#define PE_MAX_NB_SUBFR             4
#define PE_SUBFR_LENGTH_MS          5   /* 5 ms */

#define PE_LTP_MEM_LENGTH_MS        ( 4 * PE_SUBFR_LENGTH_MS )

#define PE_MAX_FRAME_LENGTH_MS      ( PE_LTP_MEM_LENGTH_MS + PE_MAX_NB_SUBFR * PE_SUBFR_LENGTH_MS )
#define PE_MAX_FRAME_LENGTH         ( PE_MAX_FRAME_LENGTH_MS * PE_MAX_FS_KHZ )
#define PE_MAX_FRAME_LENGTH_ST_1    ( PE_MAX_FRAME_LENGTH >> 2 )
#define PE_MAX_FRAME_LENGTH_ST_2    ( PE_MAX_FRAME_LENGTH >> 1 )

#define PE_MAX_LAG_MS               18           /* 18 ms -> 56 Hz */
#define PE_MIN_LAG_MS               2            /* 2 ms -> 500 Hz */
#define PE_MAX_LAG                  ( PE_MAX_LAG_MS * PE_MAX_FS_KHZ )
#define PE_MIN_LAG                  ( PE_MIN_LAG_MS * PE_MAX_FS_KHZ )

#define PE_D_SRCH_LENGTH            24

#define PE_NB_STAGE3_LAGS           5

#define PE_NB_CBKS_STAGE2           3
#define PE_NB_CBKS_STAGE2_EXT       11

#define PE_NB_CBKS_STAGE3_MAX       34
#define PE_NB_CBKS_STAGE3_MID       24
#define PE_NB_CBKS_STAGE3_MIN       16

#define PE_NB_CBKS_STAGE3_10MS      12
#define PE_NB_CBKS_STAGE2_10MS      3

#define PE_SHORTLAG_BIAS            0.2f    /* for logarithmic weighting    */
#define PE_PREVLAG_BIAS             0.2f    /* for logarithmic weighting    */
#define PE_FLATCONTOUR_BIAS         0.05f

#define SILK_PE_MIN_COMPLEX         0
#define SILK_PE_MID_COMPLEX         1
#define SILK_PE_MAX_COMPLEX         2

/* Tables for 20 ms frames */
extern const opus_int8 silk_CB_lags_stage2[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE2_EXT ];
extern const opus_int8 silk_CB_lags_stage3[ PE_MAX_NB_SUBFR ][ PE_NB_CBKS_STAGE3_MAX ];
extern const opus_int8 silk_Lag_range_stage3[ SILK_PE_MAX_COMPLEX + 1 ] [ PE_MAX_NB_SUBFR ][ 2 ];
extern const opus_int8 silk_nb_cbk_searchs_stage3[ SILK_PE_MAX_COMPLEX + 1 ];

/* Tables for 10 ms frames */
extern const opus_int8 silk_CB_lags_stage2_10_ms[ PE_MAX_NB_SUBFR >> 1][ 3 ];
extern const opus_int8 silk_CB_lags_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ 12 ];
extern const opus_int8 silk_Lag_range_stage3_10_ms[ PE_MAX_NB_SUBFR >> 1 ][ 2 ];

#endif

