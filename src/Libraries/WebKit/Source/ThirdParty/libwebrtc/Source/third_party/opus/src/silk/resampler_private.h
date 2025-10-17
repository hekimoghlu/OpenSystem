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
#ifndef SILK_RESAMPLER_PRIVATE_H
#define SILK_RESAMPLER_PRIVATE_H

#ifdef __cplusplus
extern "C" {
#endif

#include "SigProc_FIX.h"
#include "resampler_structs.h"
#include "resampler_rom.h"

/* Number of input samples to process in the inner loop */
#define RESAMPLER_MAX_BATCH_SIZE_MS             10
#define RESAMPLER_MAX_FS_KHZ                    48
#define RESAMPLER_MAX_BATCH_SIZE_IN             ( RESAMPLER_MAX_BATCH_SIZE_MS * RESAMPLER_MAX_FS_KHZ )

/* Description: Hybrid IIR/FIR polyphase implementation of resampling */
void silk_resampler_private_IIR_FIR(
    void                            *SS,            /* I/O  Resampler state             */
    opus_int16                      out[],          /* O    Output signal               */
    const opus_int16                in[],           /* I    Input signal                */
    opus_int32                      inLen           /* I    Number of input samples     */
);

/* Description: Hybrid IIR/FIR polyphase implementation of resampling */
void silk_resampler_private_down_FIR(
    void                            *SS,            /* I/O  Resampler state             */
    opus_int16                      out[],          /* O    Output signal               */
    const opus_int16                in[],           /* I    Input signal                */
    opus_int32                      inLen           /* I    Number of input samples     */
);

/* Upsample by a factor 2, high quality */
void silk_resampler_private_up2_HQ_wrapper(
    void                            *SS,            /* I/O  Resampler state (unused)    */
    opus_int16                      *out,           /* O    Output signal [ 2 * len ]   */
    const opus_int16                *in,            /* I    Input signal [ len ]        */
    opus_int32                      len             /* I    Number of input samples     */
);

/* Upsample by a factor 2, high quality */
void silk_resampler_private_up2_HQ(
    opus_int32                      *S,             /* I/O  Resampler state [ 6 ]       */
    opus_int16                      *out,           /* O    Output signal [ 2 * len ]   */
    const opus_int16                *in,            /* I    Input signal [ len ]        */
    opus_int32                      len             /* I    Number of input samples     */
);

/* Second order AR filter */
void silk_resampler_private_AR2(
    opus_int32                      S[],            /* I/O  State vector [ 2 ]          */
    opus_int32                      out_Q8[],       /* O    Output signal               */
    const opus_int16                in[],           /* I    Input signal                */
    const opus_int16                A_Q14[],        /* I    AR coefficients, Q14        */
    opus_int32                      len             /* I    Signal length               */
);

#ifdef __cplusplus
}
#endif
#endif /* SILK_RESAMPLER_PRIVATE_H */
