/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 10, 2022.
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

#include <arm_neon.h>
#include "main.h"
#include "stack_alloc.h"
#include "NSQ.h"
#include "celt/cpu_support.h"
#include "celt/arm/armcpu.h"

opus_int32 silk_noise_shape_quantizer_short_prediction_neon(const opus_int32 *buf32, const opus_int32 *coef32, opus_int order)
{
    int32x4_t coef0 = vld1q_s32(coef32);
    int32x4_t coef1 = vld1q_s32(coef32 + 4);
    int32x4_t coef2 = vld1q_s32(coef32 + 8);
    int32x4_t coef3 = vld1q_s32(coef32 + 12);

    int32x4_t a0 = vld1q_s32(buf32 - 15);
    int32x4_t a1 = vld1q_s32(buf32 - 11);
    int32x4_t a2 = vld1q_s32(buf32 - 7);
    int32x4_t a3 = vld1q_s32(buf32 - 3);

    int32x4_t b0 = vqdmulhq_s32(coef0, a0);
    int32x4_t b1 = vqdmulhq_s32(coef1, a1);
    int32x4_t b2 = vqdmulhq_s32(coef2, a2);
    int32x4_t b3 = vqdmulhq_s32(coef3, a3);

    int32x4_t c0 = vaddq_s32(b0, b1);
    int32x4_t c1 = vaddq_s32(b2, b3);

    int32x4_t d = vaddq_s32(c0, c1);

    int64x2_t e = vpaddlq_s32(d);

    int64x1_t f = vadd_s64(vget_low_s64(e), vget_high_s64(e));

    opus_int32 out = vget_lane_s32(vreinterpret_s32_s64(f), 0);

    out += silk_RSHIFT( order, 1 );

    return out;
}


opus_int32 silk_NSQ_noise_shape_feedback_loop_neon(const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef, opus_int order)
{
    opus_int32 out;
    if (order == 8)
    {
        int32x4_t a00 = vdupq_n_s32(data0[0]);
        int32x4_t a01 = vld1q_s32(data1);  /* data1[0] ... [3] */

        int32x4_t a0 = vextq_s32 (a00, a01, 3); /* data0[0] data1[0] ...[2] */
        int32x4_t a1 = vld1q_s32(data1 + 3);  /* data1[3] ... [6] */

        /*TODO: Convert these once in advance instead of once per sample, like
          silk_noise_shape_quantizer_short_prediction_neon() does.*/
        int16x8_t coef16 = vld1q_s16(coef);
        int32x4_t coef0 = vmovl_s16(vget_low_s16(coef16));
        int32x4_t coef1 = vmovl_s16(vget_high_s16(coef16));

        /*This is not bit-exact with the C version, since we do not drop the
          lower 16 bits of each multiply, but wait until the end to truncate
          precision. This is an encoder-specific calculation (and unlike
          silk_noise_shape_quantizer_short_prediction_neon(), is not meant to
          simulate what the decoder will do). We still could use vqdmulhq_s32()
          like silk_noise_shape_quantizer_short_prediction_neon() and save
          half the multiplies, but the speed difference is not large, since we
          then need two extra adds.*/
        int64x2_t b0 = vmull_s32(vget_low_s32(a0), vget_low_s32(coef0));
        int64x2_t b1 = vmlal_s32(b0, vget_high_s32(a0), vget_high_s32(coef0));
        int64x2_t b2 = vmlal_s32(b1, vget_low_s32(a1), vget_low_s32(coef1));
        int64x2_t b3 = vmlal_s32(b2, vget_high_s32(a1), vget_high_s32(coef1));

        int64x1_t c = vadd_s64(vget_low_s64(b3), vget_high_s64(b3));
        int64x1_t cS = vrshr_n_s64(c, 15);
        int32x2_t d = vreinterpret_s32_s64(cS);

        out = vget_lane_s32(d, 0);
        vst1q_s32(data1, a0);
        vst1q_s32(data1 + 4, a1);
        return out;
    }
    return silk_NSQ_noise_shape_feedback_loop_c(data0, data1, coef, order);
}
