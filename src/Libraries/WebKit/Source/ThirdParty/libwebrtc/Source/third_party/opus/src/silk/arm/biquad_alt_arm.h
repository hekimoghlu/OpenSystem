/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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
#ifndef SILK_BIQUAD_ALT_ARM_H
# define SILK_BIQUAD_ALT_ARM_H

# include "celt/arm/armcpu.h"

# if defined(OPUS_ARM_MAY_HAVE_NEON_INTR)
void silk_biquad_alt_stride2_neon(
    const opus_int16            *in,                /* I     input signal                                               */
    const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
    const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
    opus_int32                  *S,                 /* I/O   State vector [4]                                           */
    opus_int16                  *out,               /* O     output signal                                              */
    const opus_int32            len                 /* I     signal length (must be even)                               */
);

#  if !defined(OPUS_HAVE_RTCD) && defined(OPUS_ARM_PRESUME_NEON)
#   define OVERRIDE_silk_biquad_alt_stride2                   (1)
#   define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) ((void)(arch), PRESUME_NEON(silk_biquad_alt_stride2)(in, B_Q28, A_Q28, S, out, len))
#  endif
# endif

# if !defined(OVERRIDE_silk_biquad_alt_stride2)
/*Is run-time CPU detection enabled on this platform?*/
#  if defined(OPUS_HAVE_RTCD) && (defined(OPUS_ARM_MAY_HAVE_NEON_INTR) && !defined(OPUS_ARM_PRESUME_NEON_INTR))
extern void (*const SILK_BIQUAD_ALT_STRIDE2_IMPL[OPUS_ARCHMASK+1])(
        const opus_int16            *in,                /* I     input signal                                               */
        const opus_int32            *B_Q28,             /* I     MA coefficients [3]                                        */
        const opus_int32            *A_Q28,             /* I     AR coefficients [2]                                        */
        opus_int32                  *S,                 /* I/O   State vector [4]                                           */
        opus_int16                  *out,               /* O     output signal                                              */
        const opus_int32            len                 /* I     signal length (must be even)                               */
    );
#   define OVERRIDE_silk_biquad_alt_stride2                  (1)
#   define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) ((*SILK_BIQUAD_ALT_STRIDE2_IMPL[(arch)&OPUS_ARCHMASK])(in, B_Q28, A_Q28, S, out, len))
#  elif defined(OPUS_ARM_PRESUME_NEON_INTR)
#   define OVERRIDE_silk_biquad_alt_stride2                  (1)
#   define silk_biquad_alt_stride2(in, B_Q28, A_Q28, S, out, len, arch) ((void)(arch), silk_biquad_alt_stride2_neon(in, B_Q28, A_Q28, S, out, len))
#  endif
# endif

#endif /* end SILK_BIQUAD_ALT_ARM_H */
