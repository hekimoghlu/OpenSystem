/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
   @file fft_arm.h
   @brief ARM Neon Intrinsic optimizations for fft using NE10 library
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


#if !defined(FFT_ARM_H)
#define FFT_ARM_H

#include "kiss_fft.h"

#if defined(HAVE_ARM_NE10)

int opus_fft_alloc_arm_neon(kiss_fft_state *st);
void opus_fft_free_arm_neon(kiss_fft_state *st);

void opus_fft_neon(const kiss_fft_state *st,
                   const kiss_fft_cpx *fin,
                   kiss_fft_cpx *fout);

void opus_ifft_neon(const kiss_fft_state *st,
                    const kiss_fft_cpx *fin,
                    kiss_fft_cpx *fout);

#if !defined(OPUS_HAVE_RTCD)
#define OVERRIDE_OPUS_FFT (1)

#define opus_fft_alloc_arch(_st, arch) \
   ((void)(arch), opus_fft_alloc_arm_neon(_st))

#define opus_fft_free_arch(_st, arch) \
   ((void)(arch), opus_fft_free_arm_neon(_st))

#define opus_fft(_st, _fin, _fout, arch) \
   ((void)(arch), opus_fft_neon(_st, _fin, _fout))

#define opus_ifft(_st, _fin, _fout, arch) \
   ((void)(arch), opus_ifft_neon(_st, _fin, _fout))

#endif /* OPUS_HAVE_RTCD */

#endif /* HAVE_ARM_NE10 */

#endif
