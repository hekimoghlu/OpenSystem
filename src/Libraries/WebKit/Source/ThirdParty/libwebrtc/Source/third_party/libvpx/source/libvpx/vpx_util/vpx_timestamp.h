/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#ifndef VPX_VPX_UTIL_VPX_TIMESTAMP_H_
#define VPX_VPX_UTIL_VPX_TIMESTAMP_H_

#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Rational Number with an int64 numerator
typedef struct vpx_rational64 {
  int64_t num;       // fraction numerator
  int den;           // fraction denominator
} vpx_rational64_t;  // alias for struct vpx_rational64_t

static INLINE int gcd(int64_t a, int b) {
  int r;  // remainder
  assert(a >= 0);
  assert(b > 0);
  while (b != 0) {
    r = (int)(a % b);
    a = b;
    b = r;
  }

  return (int)a;
}

static INLINE void reduce_ratio(vpx_rational64_t *ratio) {
  const int denom = gcd(ratio->num, ratio->den);
  ratio->num /= denom;
  ratio->den /= denom;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // VPX_VPX_UTIL_VPX_TIMESTAMP_H_
