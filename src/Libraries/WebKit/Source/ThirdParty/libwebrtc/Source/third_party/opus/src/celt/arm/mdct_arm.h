/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 4, 2023.
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
/**
   @file arm_mdct.h
   @brief ARM Neon Intrinsic optimizations for mdct using NE10 library
 */

/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if !defined(MDCT_ARM_H)
#define MDCT_ARM_H

#include "mdct.h"

#if defined(HAVE_ARM_NE10)
/** Compute a forward MDCT and scale by 4/N, trashes the input array */
void clt_mdct_forward_neon(const mdct_lookup *l, kiss_fft_scalar *in,
                           kiss_fft_scalar * OPUS_RESTRICT out,
                           const opus_val16 *window, int overlap,
                           int shift, int stride, int arch);

void clt_mdct_backward_neon(const mdct_lookup *l, kiss_fft_scalar *in,
                            kiss_fft_scalar * OPUS_RESTRICT out,
                            const opus_val16 *window, int overlap,
                            int shift, int stride, int arch);

#if !defined(OPUS_HAVE_RTCD)
#define OVERRIDE_OPUS_MDCT (1)
#define clt_mdct_forward(_l, _in, _out, _window, _int, _shift, _stride, _arch) \
      clt_mdct_forward_neon(_l, _in, _out, _window, _int, _shift, _stride, _arch)
#define clt_mdct_backward(_l, _in, _out, _window, _int, _shift, _stride, _arch) \
      clt_mdct_backward_neon(_l, _in, _out, _window, _int, _shift, _stride, _arch)
#endif /* OPUS_HAVE_RTCD */
#endif /* HAVE_ARM_NE10 */

#endif
