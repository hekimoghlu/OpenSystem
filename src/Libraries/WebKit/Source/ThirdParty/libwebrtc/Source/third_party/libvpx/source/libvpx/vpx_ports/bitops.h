/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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
#ifndef VPX_VPX_PORTS_BITOPS_H_
#define VPX_VPX_PORTS_BITOPS_H_

#include <assert.h>

#ifdef _MSC_VER
#if defined(_M_X64) || defined(_M_IX86)
#include <intrin.h>
#define USE_MSC_INTRINSICS
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// These versions of get_lsb() and get_msb() are only valid when n != 0
// because all of the optimized versions are undefined when n == 0:
// https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html

// use GNU builtins where available.
#if defined(__GNUC__) && \
    ((__GNUC__ == 3 && __GNUC_MINOR__ >= 4) || __GNUC__ >= 4)
static INLINE int get_lsb(unsigned int n) {
  assert(n != 0);
  return __builtin_ctz(n);
}

static INLINE int get_msb(unsigned int n) {
  assert(n != 0);
  return 31 ^ __builtin_clz(n);
}
#elif defined(USE_MSC_INTRINSICS)
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)

static INLINE int get_lsb(unsigned int n) {
  unsigned long first_set_bit;  // NOLINT(runtime/int)
  _BitScanForward(&first_set_bit, n);
  return first_set_bit;
}

static INLINE int get_msb(unsigned int n) {
  unsigned long first_set_bit;
  assert(n != 0);
  _BitScanReverse(&first_set_bit, n);
  return first_set_bit;
}
#undef USE_MSC_INTRINSICS
#else
static INLINE int get_lsb(unsigned int n) {
  int i;
  assert(n != 0);
  for (i = 0; i < 32 && !(n & 1); ++i) n >>= 1;
  return i;
}

// Returns (int)floor(log2(n)). n must be > 0.
static INLINE int get_msb(unsigned int n) {
  int log = 0;
  unsigned int value = n;
  int i;

  assert(n != 0);

  for (i = 4; i >= 0; --i) {
    const int shift = (1 << i);
    const unsigned int x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  return log;
}
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_PORTS_BITOPS_H_
