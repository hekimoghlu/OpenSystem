/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
#ifndef SILK_LPC_INV_PRED_GAIN_ARM_H
# define SILK_LPC_INV_PRED_GAIN_ARM_H

# include "celt/arm/armcpu.h"

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
opus_int32 silk_LPC_inverse_pred_gain_neon(         /* O   Returns inverse prediction gain in energy domain, Q30        */
    const opus_int16            *A_Q12,             /* I   Prediction coefficients, Q12 [order]                         */
    const opus_int              order               /* I   Prediction order                                             */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_LPC_inverse_pred_gain            (1)
#   define silk_LPC_inverse_pred_gain(A_Q12, order, arch) ((void)(arch), PRESUME_NEON(silk_LPC_inverse_pred_gain)(A_Q12, order))
#  endif
# endif

# if !defined(OVERRIDE_silk_LPC_inverse_pred_gain)
/*Is run-time CPU detection enabled on this platform?*/
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern opus_int32 (*const SILK_LPC_INVERSE_PRED_GAIN_IMPL[OPUS_ARCHMASK+1])(const opus_int16 *A_Q12, const opus_int order);
#   define OVERRIDE_silk_LPC_inverse_pred_gain            (1)
#   define silk_LPC_inverse_pred_gain(A_Q12, order, arch) ((*SILK_LPC_INVERSE_PRED_GAIN_IMPL[(arch)&OPUS_ARCHMASK])(A_Q12, order))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_silk_LPC_inverse_pred_gain            (1)
#   define silk_LPC_inverse_pred_gain(A_Q12, order, arch) ((void)(arch), silk_LPC_inverse_pred_gain_neon(A_Q12, order))
#  endif
# endif

#endif /* end SILK_LPC_INV_PRED_GAIN_ARM_H */
