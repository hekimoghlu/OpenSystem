/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 10, 2023.
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
#ifndef __CLANG_HIP_STDLIB_H__

#if !defined(__HIP__) && !defined(__OPENMP_AMDGCN__)
#error "This file is for HIP and OpenMP AMDGCN device compilation only."
#endif

#if !defined(__cplusplus)

#include <limits.h>

#ifdef __OPENMP_AMDGCN__
#define __DEVICE__ static inline __attribute__((always_inline, nothrow))
#else
#define __DEVICE__ static __device__ inline __attribute__((always_inline))
#endif

__DEVICE__
int abs(int __x) {
  int __sgn = __x >> (sizeof(int) * CHAR_BIT - 1);
  return (__x ^ __sgn) - __sgn;
}
__DEVICE__
long labs(long __x) {
  long __sgn = __x >> (sizeof(long) * CHAR_BIT - 1);
  return (__x ^ __sgn) - __sgn;
}
__DEVICE__
long long llabs(long long __x) {
  long long __sgn = __x >> (sizeof(long long) * CHAR_BIT - 1);
  return (__x ^ __sgn) - __sgn;
}

#endif // !defined(__cplusplus)

#endif // #define __CLANG_HIP_STDLIB_H__
