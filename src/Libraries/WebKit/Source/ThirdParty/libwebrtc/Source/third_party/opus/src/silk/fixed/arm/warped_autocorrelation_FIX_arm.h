/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#ifndef SILK_WARPED_AUTOCORRELATION_FIX_ARM_H
# define SILK_WARPED_AUTOCORRELATION_FIX_ARM_H

# include "celt/arm/armcpu.h"

# if defined(FIXED_POINT)

#  if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void silk_warped_autocorrelation_FIX_neon(
          opus_int32                *corr,                                  /* O    Result [order + 1]                                                          */
          opus_int                  *scale,                                 /* O    Scaling of the correlation vector                                           */
    const opus_int16                *input,                                 /* I    Input data to correlate                                                     */
    const opus_int                  warping_Q16,                            /* I    Warping coefficient                                                         */
    const opus_int                  length,                                 /* I    Length of input                                                             */
    const opus_int                  order                                   /* I    Correlation order (even)                                                    */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_warped_autocorrelation_FIX (1)
#   define silk_warped_autocorrelation_FIX(corr, scale, input, warping_Q16, length, order, arch) \
    ((void)(arch), PRESUME_NEON(silk_warped_autocorrelation_FIX)(corr, scale, input, warping_Q16, length, order))
#  endif
#  endif

#  if !defined(OVERRIDE_silk_warped_autocorrelation_FIX)
/*Is run-time CPU detection enabled on this platform?*/
#   if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void (*const SILK_WARPED_AUTOCORRELATION_FIX_IMPL[OPUS_ARCHMASK+1])(opus_int32*, opus_int*, const opus_int16*, const opus_int, const opus_int, const opus_int);
#    define OVERRIDE_silk_warped_autocorrelation_FIX (1)
#    define silk_warped_autocorrelation_FIX(corr, scale, input, warping_Q16, length, order, arch) \
    ((*SILK_WARPED_AUTOCORRELATION_FIX_IMPL[(arch)&OPUS_ARCHMASK])(corr, scale, input, warping_Q16, length, order))
#   elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#    define OVERRIDE_silk_warped_autocorrelation_FIX (1)
#    define silk_warped_autocorrelation_FIX(corr, scale, input, warping_Q16, length, order, arch) \
    ((void)(arch), silk_warped_autocorrelation_FIX_neon(corr, scale, input, warping_Q16, length, order))
#   endif
#  endif

# endif /* end FIXED_POINT */

#endif /* end SILK_WARPED_AUTOCORRELATION_FIX_ARM_H */
