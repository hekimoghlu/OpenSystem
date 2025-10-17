/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 9, 2022.
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
# include "config.h"
#endif

#include "main_FIX.h"
#include "NSQ.h"
#include "SigProc_FIX.h"

#if defined(OPUS_HAVE_RTCD)

# if (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && \
 !defined(OPUS_ARM_PRESUME_NEON_INTR))

void (*const SILK_BIQUAD_ALT_STRIDE2_IMPL[OPUS_ARCHMASK + 1])(
        const opus_int16            *in,                /* I     input signal                                               */
        const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
        const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
        opus_int32                  *S,                 /* I/O   State vector [4]                                           */
        opus_int16                  *out,               /* O     output signal                                              */
        const opus_int32            len                 /* I     signal length (must be even)                               */
) = {
      silk_biquad_alt_stride2_c,    /* ARMv4 */
      silk_biquad_alt_stride2_c,    /* EDSP */
      silk_biquad_alt_stride2_c,    /* Media */
      silk_biquad_alt_stride2_neon, /* Neon */
};

opus_int32 (*const SILK_LPC_INVERSE_PRED_GAIN_IMPL[OPUS_ARCHMASK + 1])( /* O   Returns inverse prediction gain in energy domain, Q30        */
        const opus_int16            *A_Q12,                             /* I   Prediction coefficients, Q12 [order]                         */
        const opus_int              order                               /* I   Prediction order                                             */
) = {
      silk_LPC_inverse_pred_gain_c,    /* ARMv4 */
      silk_LPC_inverse_pred_gain_c,    /* EDSP */
      silk_LPC_inverse_pred_gain_c,    /* Media */
      silk_LPC_inverse_pred_gain_neon, /* Neon */
};

void  (*const SILK_NSQ_DEL_DEC_IMPL[OPUS_ARCHMASK + 1])(
        const silk_encoder_state    *psEncC,                                    /* I    Encoder State                   */
        silk_nsq_state              *NSQ,                                       /* I/O  NSQ state                       */
        SideInfoIndices             *psIndices,                                 /* I/O  Quantization Indices            */
        const opus_int16            x16[],                                      /* I    Input                           */
        opus_int8                   pulses[],                                   /* O    Quantized pulse signal          */
        const opus_int16            PredCoef_Q12[ 2 * MAX_LPC_ORDER ],          /* I    Short term prediction coefs     */
        const opus_int16            LTPCoef_Q14[ LTP_ORDER * MAX_NB_SUBFR ],    /* I    Long term prediction coefs      */
        const opus_int16            AR_Q13[ MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER ], /* I Noise shaping coefs              */
        const opus_int              HarmShapeGain_Q14[ MAX_NB_SUBFR ],          /* I    Long term shaping coefs         */
        const opus_int              Tilt_Q14[ MAX_NB_SUBFR ],                   /* I    Spectral tilt                   */
        const opus_int32            LF_shp_Q14[ MAX_NB_SUBFR ],                 /* I    Low frequency shaping coefs     */
        const opus_int32            Gains_Q16[ MAX_NB_SUBFR ],                  /* I    Quantization step sizes         */
        const opus_int              pitchL[ MAX_NB_SUBFR ],                     /* I    Pitch lags                      */
        const opus_int              Lambda_Q10,                                 /* I    Rate/distortion tradeoff        */
        const opus_int              LTP_scale_Q14                               /* I    LTP state scaling               */
) = {
      silk_NSQ_del_dec_c,    /* ARMv4 */
      silk_NSQ_del_dec_c,    /* EDSP */
      silk_NSQ_del_dec_c,    /* Media */
      silk_NSQ_del_dec_neon, /* Neon */
};

/*There is no table for silk_noise_shape_quantizer_short_prediction because the
   NEON version takes different parameters than the C version.
  Instead RTCD is done via if statements at the call sites.
  See NSQ_neon.h for details.*/

opus_int32
 (*const SILK_NSQ_NOISE_SHAPE_FEEDBACK_LOOP_IMPL[OPUS_ARCHMASK+1])(
 const opus_int32 *data0, opus_int32 *data1, const opus_int16 *coef,
 opus_int order) = {
  silk_NSQ_noise_shape_feedback_loop_c,    /* ARMv4 */
  silk_NSQ_noise_shape_feedback_loop_c,    /* EDSP */
  silk_NSQ_noise_shape_feedback_loop_c,    /* Media */
  silk_NSQ_noise_shape_feedback_loop_neon, /* NEON */
};

# endif

# if defined(FIXED_POINT) && \
 defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR)

void (*const SILK_WARPED_AUTOCORRELATION_FIX_IMPL[OPUS_ARCHMASK + 1])(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
) = {
      silk_warped_autocorrelation_FIX_c,    /* ARMv4 */
      silk_warped_autocorrelation_FIX_c,    /* EDSP */
      silk_warped_autocorrelation_FIX_c,    /* Media */
      silk_warped_autocorrelation_FIX_neon, /* Neon */
};

# endif

#endif /* OPUS_HAVE_RTCD */
