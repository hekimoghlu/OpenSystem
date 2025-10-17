/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 29, 2024.
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
#ifndef SILK_PLC_H
#define SILK_PLC_H

#include "main.h"

#define BWE_COEF                        0.99
#define V_PITCH_GAIN_START_MIN_Q14      11469               /* 0.7 in Q14               */
#define V_PITCH_GAIN_START_MAX_Q14      15565               /* 0.95 in Q14              */
#define MAX_PITCH_LAG_MS                18
#define RAND_BUF_SIZE                   128
#define RAND_BUF_MASK                   ( RAND_BUF_SIZE - 1 )
#define LOG2_INV_LPC_GAIN_HIGH_THRES    3                   /* 2^3 = 8 dB LPC gain      */
#define LOG2_INV_LPC_GAIN_LOW_THRES     8                   /* 2^8 = 24 dB LPC gain     */
#define PITCH_DRIFT_FAC_Q16             655                 /* 0.01 in Q16              */

void silk_PLC_Reset(
    silk_decoder_state                  *psDec              /* I/O Decoder state        */
);

void silk_PLC(
    silk_decoder_state                  *psDec,             /* I/O Decoder state        */
    silk_decoder_control                *psDecCtrl,         /* I/O Decoder control      */
    opus_int16                          frame[],            /* I/O  signal              */
    opus_int                            lost,               /* I Loss flag              */
    int                                 arch                /* I Run-time architecture  */
);

void silk_PLC_glue_frames(
    silk_decoder_state                  *psDec,             /* I/O decoder state        */
    opus_int16                          frame[],            /* I/O signal               */
    opus_int                            length              /* I length of signal       */
);

#endif

