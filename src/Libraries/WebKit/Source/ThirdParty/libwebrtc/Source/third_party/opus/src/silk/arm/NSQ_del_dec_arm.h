/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 15, 2022.
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
#ifndef SILK_NSQ_DEL_DEC_ARM_H
#define SILK_NSQ_DEL_DEC_ARM_H

#include "celt/arm/armcpu.h"

#if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void silk_NSQ_del_dec_neon(
    const silk_encoder_state *psEncC, silk_nsq_state *NSQ,
    SideInfoIndices *psIndices, const opus_int16 x16[], opus_int8 pulses[],
    const opus_int16 PredCoef_Q12[2 * MAX_LPC_ORDER],
    const opus_int16 LTPCoef_Q14[LTP_ORDER * MAX_NB_SUBFR],
    const opus_int16 AR_Q13[MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    const opus_int HarmShapeGain_Q14[MAX_NB_SUBFR],
    const opus_int Tilt_Q14[MAX_NB_SUBFR],
    const opus_int32 LF_shp_Q14[MAX_NB_SUBFR],
    const opus_int32 Gains_Q16[MAX_NB_SUBFR],
    const opus_int pitchL[MAX_NB_SUBFR], const opus_int Lambda_Q10,
    const opus_int LTP_scale_Q14);

#if !defined(OPUS_HAVE_RTCD)
#define OVERRIDE_silk_NSQ_del_dec (1)
#define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12,  \
                         LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14, Tilt_Q14,   \
                         LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10,          \
                         LTP_scale_Q14, arch)                                \
    ((void)(arch),                                                           \
     PRESUME_NEON(silk_NSQ_del_dec)(                                         \
         psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14,     \
         AR_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, \
         Lambda_Q10, LTP_scale_Q14))
#endif
#endif

#if !defined(OVERRIDE_silk_NSQ_del_dec)
/*Is run-time CPU detection enabled on this platform?*/
#if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && \
                                !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void (*const SILK_NSQ_DEL_DEC_IMPL[OPUS_ARCHMASK + 1])(
    const silk_encoder_state *psEncC, silk_nsq_state *NSQ,
    SideInfoIndices *psIndices, const opus_int16 x16[], opus_int8 pulses[],
    const opus_int16 PredCoef_Q12[2 * MAX_LPC_ORDER],
    const opus_int16 LTPCoef_Q14[LTP_ORDER * MAX_NB_SUBFR],
    const opus_int16 AR_Q13[MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    const opus_int HarmShapeGain_Q14[MAX_NB_SUBFR],
    const opus_int Tilt_Q14[MAX_NB_SUBFR],
    const opus_int32 LF_shp_Q14[MAX_NB_SUBFR],
    const opus_int32 Gains_Q16[MAX_NB_SUBFR],
    const opus_int pitchL[MAX_NB_SUBFR], const opus_int Lambda_Q10,
    const opus_int LTP_scale_Q14);
#define OVERRIDE_silk_NSQ_del_dec (1)
#define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, \
                         LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14, Tilt_Q14,  \
                         LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10,         \
                         LTP_scale_Q14, arch)                               \
    ((*SILK_NSQ_DEL_DEC_IMPL[(arch)&OPUS_ARCHMASK])(                        \
        psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, LTPCoef_Q14,     \
        AR_Q13, HarmShapeGain_Q14, Tilt_Q14, LF_shp_Q14, Gains_Q16, pitchL, \
        Lambda_Q10, LTP_scale_Q14))
#elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#define OVERRIDE_silk_NSQ_del_dec (1)
#define silk_NSQ_del_dec(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12,   \
                         LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14, Tilt_Q14,    \
                         LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10,           \
                         LTP_scale_Q14, arch)                                 \
    ((void)(arch),                                                            \
     silk_NSQ_del_dec_neon(psEncC, NSQ, psIndices, x16, pulses, PredCoef_Q12, \
                           LTPCoef_Q14, AR_Q13, HarmShapeGain_Q14, Tilt_Q14,  \
                           LF_shp_Q14, Gains_Q16, pitchL, Lambda_Q10,         \
                           LTP_scale_Q14))
#endif
#endif

#endif /* end SILK_NSQ_DEL_DEC_ARM_H */
