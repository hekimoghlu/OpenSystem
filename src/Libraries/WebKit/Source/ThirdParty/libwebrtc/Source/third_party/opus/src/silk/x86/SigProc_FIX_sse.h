/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 28, 2022.
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
#ifndef SIGPROC_FIX_SSE_H
# define SIGPROC_FIX_SSE_H

# ifdef HAVE_CONFIG_H
#  include "config.h"
# endif

# if defined(OPUS_X86_MAY_HAVE_SSE4_1)
void silk_burg_modified_sse4_1(
    opus_int32                  *res_nrg,           /* O    Residual energy                                             */
    opus_int                    *res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */
);

#  if defined(OPUS_X86_PRESUME_SSE4_1)

#   define OVERRIDE_silk_burg_modified
#   define silk_burg_modified(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch) \
       ((void)(arch), silk_burg_modified_sse4_1(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch))

#  elif defined(OPUS_HAVE_RTCD)

extern void (*const SILK_BURG_MODIFIED_IMPL[OPUS_ARCHMASK + 1])(
    opus_int32                  *res_nrg,           /* O    Residual energy                                             */
    opus_int                    *res_nrg_Q,         /* O    Residual energy Q value                                     */
    opus_int32                  A_Q16[],            /* O    Prediction coefficients (length order)                      */
    const opus_int16            x[],                /* I    Input signal, length: nb_subfr * ( D + subfr_length )       */
    const opus_int32            minInvGain_Q30,     /* I    Inverse of max prediction gain                              */
    const opus_int              subfr_length,       /* I    Input signal subframe length (incl. D preceding samples)    */
    const opus_int              nb_subfr,           /* I    Number of subframes stacked in x                            */
    const opus_int              D,                  /* I    Order                                                       */
    int                         arch                /* I    Run-time architecture                                       */);

#   define OVERRIDE_silk_burg_modified
#   define silk_burg_modified(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch) \
     ((*SILK_BURG_MODIFIED_IMPL[(arch) & OPUS_ARCHMASK])(res_nrg, res_nrg_Q, A_Q16, x, minInvGain_Q30, subfr_length, nb_subfr, D, arch))

#  endif

opus_int64 silk_inner_prod16_sse4_1(
    const opus_int16 *inVec1,
    const opus_int16 *inVec2,
    const opus_int   len
);


#  if defined(OPUS_X86_PRESUME_SSE4_1)

#   define OVERRIDE_silk_inner_prod16
#   define silk_inner_prod16(inVec1, inVec2, len, arch) \
       ((void)(arch),silk_inner_prod16_sse4_1(inVec1, inVec2, len))

#  elif defined(OPUS_HAVE_RTCD)

extern opus_int64 (*const SILK_INNER_PROD16_IMPL[OPUS_ARCHMASK + 1])(
                    const opus_int16 *inVec1,
                    const opus_int16 *inVec2,
                    const opus_int   len);

#   define OVERRIDE_silk_inner_prod16
#   define silk_inner_prod16(inVec1, inVec2, len, arch) \
     ((*SILK_INNER_PROD16_IMPL[(arch) & OPUS_ARCHMASK])(inVec1, inVec2, len))

#  endif
# endif
#endif
