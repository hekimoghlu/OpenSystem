/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 25, 2023.
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
#ifndef SILK_MACROS_ARMv4_H
#define SILK_MACROS_ARMv4_H

/* This macro only avoids the undefined behaviour from a left shift of
   a negative value. It should only be used in macros that can't include
   SigProc_FIX.h. In other cases, use silk_LSHIFT32(). */
#define SAFE_SHL(a,b) ((opus_int32)((opus_uint32)(a) << (b)))

/* (a32 * (opus_int32)((opus_int16)(b32))) >> 16 output have to be 32bit int */
#undef silk_SMULWB
static OPUS_INLINE opus_int32 silk_SMULWB_armv4(opus_int32 a, opus_int16 b)
{
  unsigned rd_lo;
  int rd_hi;
  __asm__(
      "#silk_SMULWB\n\t"
      "smull %0, %1, %2, %3\n\t"
      : "=&r"(rd_lo), "=&r"(rd_hi)
      : "%r"(a), "r"(SAFE_SHL(b,16))
  );
  return rd_hi;
}
#define silk_SMULWB(a, b) (silk_SMULWB_armv4(a, b))

/* a32 + (b32 * (opus_int32)((opus_int16)(c32))) >> 16 output have to be 32bit int */
#undef silk_SMLAWB
#define silk_SMLAWB(a, b, c) ((a) + silk_SMULWB(b, c))

/* (a32 * (b32 >> 16)) >> 16 */
#undef silk_SMULWT
static OPUS_INLINE opus_int32 silk_SMULWT_armv4(opus_int32 a, opus_int32 b)
{
  unsigned rd_lo;
  int rd_hi;
  __asm__(
      "#silk_SMULWT\n\t"
      "smull %0, %1, %2, %3\n\t"
      : "=&r"(rd_lo), "=&r"(rd_hi)
      : "%r"(a), "r"(b&~0xFFFF)
  );
  return rd_hi;
}
#define silk_SMULWT(a, b) (silk_SMULWT_armv4(a, b))

/* a32 + (b32 * (c32 >> 16)) >> 16 */
#undef silk_SMLAWT
#define silk_SMLAWT(a, b, c) ((a) + silk_SMULWT(b, c))

/* (a32 * b32) >> 16 */
#undef silk_SMULWW
static OPUS_INLINE opus_int32 silk_SMULWW_armv4(opus_int32 a, opus_int32 b)
{
  unsigned rd_lo;
  int rd_hi;
  __asm__(
    "#silk_SMULWW\n\t"
    "smull %0, %1, %2, %3\n\t"
    : "=&r"(rd_lo), "=&r"(rd_hi)
    : "%r"(a), "r"(b)
  );
  return SAFE_SHL(rd_hi,16)+(rd_lo>>16);
}
#define silk_SMULWW(a, b) (silk_SMULWW_armv4(a, b))

#undef silk_SMLAWW
static OPUS_INLINE opus_int32 silk_SMLAWW_armv4(opus_int32 a, opus_int32 b,
 opus_int32 c)
{
  unsigned rd_lo;
  int rd_hi;
  __asm__(
    "#silk_SMLAWW\n\t"
    "smull %0, %1, %2, %3\n\t"
    : "=&r"(rd_lo), "=&r"(rd_hi)
    : "%r"(b), "r"(c)
  );
  return a+SAFE_SHL(rd_hi,16)+(rd_lo>>16);
}
#define silk_SMLAWW(a, b, c) (silk_SMLAWW_armv4(a, b, c))

#undef SAFE_SHL

#endif /* SILK_MACROS_ARMv4_H */
