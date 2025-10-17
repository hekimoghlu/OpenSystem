/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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
   @file fixed_c5x.h
   @brief Fixed-point operations for the TI C5x DSP family
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

#ifndef FIXED_C5X_H
#define FIXED_C5X_H

#include "dsplib.h"

#undef IMUL32
static OPUS_INLINE long IMUL32(long i, long j)
{
   long ac0, ac1;
   ac0 = _lmpy(i>>16,j);
   ac1 = ac0 + _lmpy(i,j>>16);
   return _lmpyu(i,j) + (ac1<<16);
}

#undef MAX16
#define MAX16(a,b) _max(a,b)

#undef MIN16
#define MIN16(a,b) _min(a,b)

#undef MAX32
#define MAX32(a,b) _lmax(a,b)

#undef MIN32
#define MIN32(a,b) _lmin(a,b)

#undef VSHR32
#define VSHR32(a, shift) _lshl(a,-(shift))

#undef MULT16_16_Q15
#define MULT16_16_Q15(a,b) (_smpy(a,b))

#undef MULT16_16SU
#define MULT16_16SU(a,b) _lmpysu(a,b)

#undef MULT_16_16
#define MULT_16_16(a,b) _lmpy(a,b)

/* FIXME: This is technically incorrect and is bound to cause problems. Is there any cleaner solution? */
#undef MULT16_32_Q15
#define MULT16_32_Q15(a,b) ADD32(SHL(MULT16_16((a),SHR((b),16)),1), SHR(MULT16_16SU((a),(b)),15))

#define celt_ilog2(x) (30 - _lnorm(x))
#define OVERRIDE_CELT_ILOG2

#define celt_maxabs16(x, len) MAX32(EXTEND32(maxval((DATA *)x, len)),-EXTEND32(minval((DATA *)x, len)))
#define OVERRIDE_CELT_MAXABS16

#endif /* FIXED_C5X_H */
