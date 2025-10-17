/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 23, 2022.
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
#ifndef SILK_SIGPROC_FIX_ARMv5E_H
#define SILK_SIGPROC_FIX_ARMv5E_H

#undef silk_SMULTT
static OPUS_INLINE opus_int32 silk_SMULTT_armv5e(opus_int32 a, opus_int32 b)
{
  opus_int32 res;
  __asm__(
      "#silk_SMULTT\n\t"
      "smultt %0, %1, %2\n\t"
      : "=r"(res)
      : "%r"(a), "r"(b)
  );
  return res;
}
#define silk_SMULTT(a, b) (silk_SMULTT_armv5e(a, b))

#undef silk_SMLATT
static OPUS_INLINE opus_int32 silk_SMLATT_armv5e(opus_int32 a, opus_int32 b,
 opus_int32 c)
{
  opus_int32 res;
  __asm__(
      "#silk_SMLATT\n\t"
      "smlatt %0, %1, %2, %3\n\t"
      : "=r"(res)
      : "%r"(b), "r"(c), "r"(a)
  );
  return res;
}
#define silk_SMLATT(a, b, c) (silk_SMLATT_armv5e(a, b, c))

#endif
