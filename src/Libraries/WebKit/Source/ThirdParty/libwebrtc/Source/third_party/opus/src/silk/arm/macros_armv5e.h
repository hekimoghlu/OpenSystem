/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#ifndef SILK_MACROS_ARMv5E_H
#define SILK_MACROS_ARMv5E_H

/* This macro only avoids the undefined behaviour from a left shift of
   a negative value. It should only be used in macros that can't include
   SigProc_FIX.h. In other cases, use silk_LSHIFT32(). */
#define SAFE_SHL(a,b) ((opus_int32)((opus_uint32)(a) << (b)))

/* (a32 * (opus_int32)((opus_int16)(b32))) >> 16 output have to be 32bit int */
#undef silk_SMULWB
static OPUS_INLINE opus_int32 silk_SMULWB_armv5e(opus_int32 a, opus_int16 b)
{
  int res;
  __asm__(
      "#silk_SMULWB\n\t"
      "smulwb %0, %1, %2\n\t"
      : "=r"(res)
      : "r"(a), "r"(b)
  );
  return res;
}
#define silk_SMULWB(a, b) (silk_SMULWB_armv5e(a, b))

/* a32 + (b32 * (opus_int32)((opus_int16)(c32))) >> 16 output have to be 32bit int */
#undef silk_SMLAWB
static OPUS_INLINE opus_int32 silk_SMLAWB_armv5e(opus_int32 a, opus_int32 b,
 opus_int16 c)
{
  int res;
  __asm__(
      "#silk_SMLAWB\n\t"
      "smlawb %0, %1, %2, %3\n\t"
      : "=r"(res)
      : "r"(b), "r"(c), "r"(a)
  );
  return res;
}
#define silk_SMLAWB(a, b, c) (silk_SMLAWB_armv5e(a, b, c))

/* (a32 * (b32 >> 16)) >> 16 */
#undef silk_SMULWT
static OPUS_INLINE opus_int32 silk_SMULWT_armv5e(opus_int32 a, opus_int32 b)
{
  int res;
  __asm__(
      "#silk_SMULWT\n\t"
      "smulwt %0, %1, %2\n\t"
      : "=r"(res)
      : "r"(a), "r"(b)
  );
  return res;
}
#define silk_SMULWT(a, b) (silk_SMULWT_armv5e(a, b))

/* a32 + (b32 * (c32 >> 16)) >> 16 */
#undef silk_SMLAWT
static OPUS_INLINE opus_int32 silk_SMLAWT_armv5e(opus_int32 a, opus_int32 b,
 opus_int32 c)
{
  int res;
  __asm__(
      "#silk_SMLAWT\n\t"
      "smlawt %0, %1, %2, %3\n\t"
      : "=r"(res)
      : "r"(b), "r"(c), "r"(a)
  );
  return res;
}
#define silk_SMLAWT(a, b, c) (silk_SMLAWT_armv5e(a, b, c))

/* (opus_int32)((opus_int16)(a3))) * (opus_int32)((opus_int16)(b32)) output have to be 32bit int */
#undef silk_SMULBB
static OPUS_INLINE opus_int32 silk_SMULBB_armv5e(opus_int32 a, opus_int32 b)
{
  int res;
  __asm__(
      "#silk_SMULBB\n\t"
      "smulbb %0, %1, %2\n\t"
      : "=r"(res)
      : "%r"(a), "r"(b)
  );
  return res;
}
#define silk_SMULBB(a, b) (silk_SMULBB_armv5e(a, b))

/* a32 + (opus_int32)((opus_int16)(b32)) * (opus_int32)((opus_int16)(c32)) output have to be 32bit int */
#undef silk_SMLABB
static OPUS_INLINE opus_int32 silk_SMLABB_armv5e(opus_int32 a, opus_int32 b,
 opus_int32 c)
{
  int res;
  __asm__(
      "#silk_SMLABB\n\t"
      "smlabb %0, %1, %2, %3\n\t"
      : "=r"(res)
      : "%r"(b), "r"(c), "r"(a)
  );
  return res;
}
#define silk_SMLABB(a, b, c) (silk_SMLABB_armv5e(a, b, c))

/* (opus_int32)((opus_int16)(a32)) * (b32 >> 16) */
#undef silk_SMULBT
static OPUS_INLINE opus_int32 silk_SMULBT_armv5e(opus_int32 a, opus_int32 b)
{
  int res;
  __asm__(
      "#silk_SMULBT\n\t"
      "smulbt %0, %1, %2\n\t"
      : "=r"(res)
      : "r"(a), "r"(b)
  );
  return res;
}
#define silk_SMULBT(a, b) (silk_SMULBT_armv5e(a, b))

/* a32 + (opus_int32)((opus_int16)(b32)) * (c32 >> 16) */
#undef silk_SMLABT
static OPUS_INLINE opus_int32 silk_SMLABT_armv5e(opus_int32 a, opus_int32 b,
 opus_int32 c)
{
  int res;
  __asm__(
      "#silk_SMLABT\n\t"
      "smlabt %0, %1, %2, %3\n\t"
      : "=r"(res)
      : "r"(b), "r"(c), "r"(a)
  );
  return res;
}
#define silk_SMLABT(a, b, c) (silk_SMLABT_armv5e(a, b, c))

/* add/subtract with output saturated */
#undef silk_ADD_SAT32
static OPUS_INLINE opus_int32 silk_ADD_SAT32_armv5e(opus_int32 a, opus_int32 b)
{
  int res;
  __asm__(
      "#silk_ADD_SAT32\n\t"
      "qadd %0, %1, %2\n\t"
      : "=r"(res)
      : "%r"(a), "r"(b)
  );
  return res;
}
#define silk_ADD_SAT32(a, b) (silk_ADD_SAT32_armv5e(a, b))

#undef silk_SUB_SAT32
static OPUS_INLINE opus_int32 silk_SUB_SAT32_armv5e(opus_int32 a, opus_int32 b)
{
  int res;
  __asm__(
      "#silk_SUB_SAT32\n\t"
      "qsub %0, %1, %2\n\t"
      : "=r"(res)
      : "r"(a), "r"(b)
  );
  return res;
}
#define silk_SUB_SAT32(a, b) (silk_SUB_SAT32_armv5e(a, b))

#undef silk_CLZ16
static OPUS_INLINE opus_int32 silk_CLZ16_armv5(opus_int16 in16)
{
  int res;
  __asm__(
      "#silk_CLZ16\n\t"
      "clz %0, %1;\n"
      : "=r"(res)
      : "r"(SAFE_SHL(in16,16)|0x8000)
  );
  return res;
}
#define silk_CLZ16(in16) (silk_CLZ16_armv5(in16))

#undef silk_CLZ32
static OPUS_INLINE opus_int32 silk_CLZ32_armv5(opus_int32 in32)
{
  int res;
  __asm__(
      "#silk_CLZ32\n\t"
      "clz %0, %1\n\t"
      : "=r"(res)
      : "r"(in32)
  );
  return res;
}
#define silk_CLZ32(in32) (silk_CLZ32_armv5(in32))

#undef SAFE_SHL

#endif /* SILK_MACROS_ARMv5E_H */
