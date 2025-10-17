/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#ifndef SILK_NSQ_NEON_H
#define SILK_NSQ_NEON_H

#include "cpu_support.h"
#include "SigProc_FIX.h"

#undef silk_short_prediction_create_arch_coef
/* For vectorized calc, reverse a_Q12 coefs, convert to 32-bit, and shift for vqdmulhq_s32. */
static OPUS_INLINE void silk_short_prediction_create_arch_coef_neon(opus_int32 *out, const opus_int16 *in, opus_int order)
{
    out[15] = silk_LSHIFT32(in[0], 15);
    out[14] = silk_LSHIFT32(in[1], 15);
    out[13] = silk_LSHIFT32(in[2], 15);
    out[12] = silk_LSHIFT32(in[3], 15);
    out[11] = silk_LSHIFT32(in[4], 15);
    out[10] = silk_LSHIFT32(in[5], 15);
    out[9]  = silk_LSHIFT32(in[6], 15);
    out[8]  = silk_LSHIFT32(in[7], 15);
    out[7]  = silk_LSHIFT32(in[8], 15);
    out[6]  = silk_LSHIFT32(in[9], 15);

    if (order == 16)
    {
        out[5] = silk_LSHIFT32(in[10], 15);
        out[4] = silk_LSHIFT32(in[11], 15);
        out[3] = silk_LSHIFT32(in[12], 15);
        out[2] = silk_LSHIFT32(in[13], 15);
        out[1] = silk_LSHIFT32(in[14], 15);
        out[0] = silk_LSHIFT32(in[15], 15);
    }
    else
    {
        out[5] = 0;
        out[4] = 0;
        out[3] = 0;
        out[2] = 0;
        out[1] = 0;
        out[0] = 0;
    }
}

#if defined(OPUS_ARM_PRESUME_NEON_INTR)

#define silk_short_prediction_create_arch_coef(out, in, order) \
    (silk_short_prediction_create_arch_coef_neon(out, in, order))

#elif defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

#define silk_short_prediction_create_arch_coef(out, in, order) \
    do { if (arch == OPUS_ARCH_ARM_NEON) { silk_short_prediction_create_arch_coef_neon(out, in, order); } } while (0)

#endif

opus_int32 silk_noise_shape_quantizer_short_prediction_neon(const opus_int32 *buf32, const opus_int32 *coef32, opus_int order);

opus_int32 silk_NSQ_noise_shape_feedback_loop_neon(const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef, opus_int order);

#if defined(OPUS_ARM_PRESUME_NEON_INTR)
#undef silk_noise_shape_quantizer_short_prediction
#define silk_noise_shape_quantizer_short_prediction(in, coef, coefRev, order, arch) \
    ((void)arch,silk_noise_shape_quantizer_short_prediction_neon(in, coefRev, order))

#undef silk_NSQ_noise_shape_feedback_loop
#define silk_NSQ_noise_shape_feedback_loop(data0, data1, coef, order, arch)  ((void)arch,silk_NSQ_noise_shape_feedback_loop_neon(data0, data1, coef, order))

#elif defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_MAY_HAVE_NEON_INTR)

/* silk_noise_shape_quantizer_short_prediction implementations take different parameters based on arch
   (coef vs. coefRev) so can't use the usual IMPL table implementation */
#undef silk_noise_shape_quantizer_short_prediction
#define silk_noise_shape_quantizer_short_prediction(in, coef, coefRev, order, arch)  \
    (arch == OPUS_ARCH_ARM_NEON ? \
        silk_noise_shape_quantizer_short_prediction_neon(in, coefRev, order) : \
        silk_noise_shape_quantizer_short_prediction_c(in, coef, order))

extern opus_int32
 (*const SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[OPUS_ARCHMASK+1])(
 const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef,
 opus_int order);

#undef silk_NSQ_noise_shape_feedback_loop
#define silk_NSQ_noise_shape_feedback_loop(data0, data1, coef, order, arch) \
 (SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[(arch)&OPUS_ARCHMASK](data0, data1, \
 coef, order))

#endif

#endif /* SILK_NSQ_NEON_H */
