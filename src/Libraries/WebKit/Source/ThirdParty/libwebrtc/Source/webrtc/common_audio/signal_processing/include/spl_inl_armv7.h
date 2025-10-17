/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
/* This header file includes the inline functions for ARM processors in
 * the fix point signal processing library.
 */

#ifndef COMMON_AUDIO_SIGNAL_PROCESSING_INCLUDE_SPL_INL_ARMV7_H_
#define COMMON_AUDIO_SIGNAL_PROCESSING_INCLUDE_SPL_INL_ARMV7_H_

#include <stdint.h>

/* TODO(kma): Replace some assembly code with GCC intrinsics
 * (e.g. __builtin_clz).
 */

/* This function produces result that is not bit exact with that by the generic
 * C version in some cases, although the former is at least as accurate as the
 * later.
 */
static __inline int32_t WEBRTC_SPL_MUL_16_32_RSFT16(int16_t a, int32_t b) {
  int32_t tmp = 0;
  __asm __volatile("smulwb %0, %1, %2" : "=r"(tmp) : "r"(b), "r"(a));
  return tmp;
}

static __inline int32_t WEBRTC_SPL_MUL_16_16(int16_t a, int16_t b) {
  int32_t tmp = 0;
  __asm __volatile("smulbb %0, %1, %2" : "=r"(tmp) : "r"(a), "r"(b));
  return tmp;
}

// TODO(kma): add unit test.
static __inline int32_t WebRtc_MulAccumW16(int16_t a, int16_t b, int32_t c) {
  int32_t tmp = 0;
  __asm __volatile("smlabb %0, %1, %2, %3"
                   : "=r"(tmp)
                   : "r"(a), "r"(b), "r"(c));
  return tmp;
}

static __inline int16_t WebRtcSpl_AddSatW16(int16_t a, int16_t b) {
  int32_t s_sum = 0;

  __asm __volatile("qadd16 %0, %1, %2" : "=r"(s_sum) : "r"(a), "r"(b));

  return (int16_t)s_sum;
}

static __inline int32_t WebRtcSpl_AddSatW32(int32_t l_var1, int32_t l_var2) {
  int32_t l_sum = 0;

  __asm __volatile("qadd %0, %1, %2" : "=r"(l_sum) : "r"(l_var1), "r"(l_var2));

  return l_sum;
}

static __inline int32_t WebRtcSpl_SubSatW32(int32_t l_var1, int32_t l_var2) {
  int32_t l_sub = 0;

  __asm __volatile("qsub %0, %1, %2" : "=r"(l_sub) : "r"(l_var1), "r"(l_var2));

  return l_sub;
}

static __inline int16_t WebRtcSpl_SubSatW16(int16_t var1, int16_t var2) {
  int32_t s_sub = 0;

  __asm __volatile("qsub16 %0, %1, %2" : "=r"(s_sub) : "r"(var1), "r"(var2));

  return (int16_t)s_sub;
}

static __inline int16_t WebRtcSpl_GetSizeInBits(uint32_t n) {
  int32_t tmp = 0;

  __asm __volatile("clz %0, %1" : "=r"(tmp) : "r"(n));

  return (int16_t)(32 - tmp);
}

static __inline int16_t WebRtcSpl_NormW32(int32_t a) {
  int32_t tmp = 0;

  if (a == 0) {
    return 0;
  } else if (a < 0) {
    a ^= 0xFFFFFFFF;
  }

  __asm __volatile("clz %0, %1" : "=r"(tmp) : "r"(a));

  return (int16_t)(tmp - 1);
}

static __inline int16_t WebRtcSpl_NormU32(uint32_t a) {
  int tmp = 0;

  if (a == 0)
    return 0;

  __asm __volatile("clz %0, %1" : "=r"(tmp) : "r"(a));

  return (int16_t)tmp;
}

static __inline int16_t WebRtcSpl_NormW16(int16_t a) {
  int32_t tmp = 0;
  int32_t a_32 = a;

  if (a_32 == 0) {
    return 0;
  } else if (a_32 < 0) {
    a_32 ^= 0xFFFFFFFF;
  }

  __asm __volatile("clz %0, %1" : "=r"(tmp) : "r"(a_32));

  return (int16_t)(tmp - 17);
}

// TODO(kma): add unit test.
static __inline int16_t WebRtcSpl_SatW32ToW16(int32_t value32) {
  int32_t out = 0;

  __asm __volatile("ssat %0, #16, %1" : "=r"(out) : "r"(value32));

  return (int16_t)out;
}

#endif  // COMMON_AUDIO_SIGNAL_PROCESSING_INCLUDE_SPL_INL_ARMV7_H_
