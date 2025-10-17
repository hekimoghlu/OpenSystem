/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// The default numeric_limits<T> template shall have all members, but with
// 0 or false values.

#include <uscl/std/cassert>
#include <uscl/std/limits>

#include "test_macros.h"

struct A
{
  __host__ __device__ A(int i = 0)
      : data_(i)
  {}
  int data_;
};

__host__ __device__ bool operator==(const A& x, const A& y)
{
  return x.data_ == y.data_;
}

int main(int, char**)
{
  static_assert(cuda::std::numeric_limits<A>::is_specialized == false,
                "cuda::std::numeric_limits<A>::is_specialized == false");
  assert(cuda::std::numeric_limits<A>::min() == A());
  assert(cuda::std::numeric_limits<A>::max() == A());
  assert(cuda::std::numeric_limits<A>::lowest() == A());
  static_assert(cuda::std::numeric_limits<A>::digits == 0, "cuda::std::numeric_limits<A>::digits == 0");
  static_assert(cuda::std::numeric_limits<A>::digits10 == 0, "cuda::std::numeric_limits<A>::digits10 == 0");
  static_assert(cuda::std::numeric_limits<A>::max_digits10 == 0, "cuda::std::numeric_limits<A>::max_digits10 == 0");
  static_assert(cuda::std::numeric_limits<A>::is_signed == false, "cuda::std::numeric_limits<A>::is_signed == false");
  static_assert(cuda::std::numeric_limits<A>::is_integer == false, "cuda::std::numeric_limits<A>::is_integer == false");
  static_assert(cuda::std::numeric_limits<A>::is_exact == false, "cuda::std::numeric_limits<A>::is_exact == false");
  static_assert(cuda::std::numeric_limits<A>::radix == 0, "cuda::std::numeric_limits<A>::radix == 0");
  assert(cuda::std::numeric_limits<A>::epsilon() == A());
  assert(cuda::std::numeric_limits<A>::round_error() == A());
  static_assert(cuda::std::numeric_limits<A>::min_exponent == 0, "cuda::std::numeric_limits<A>::min_exponent == 0");
  static_assert(cuda::std::numeric_limits<A>::min_exponent10 == 0, "cuda::std::numeric_limits<A>::min_exponent10 == 0");
  static_assert(cuda::std::numeric_limits<A>::max_exponent == 0, "cuda::std::numeric_limits<A>::max_exponent == 0");
  static_assert(cuda::std::numeric_limits<A>::max_exponent10 == 0, "cuda::std::numeric_limits<A>::max_exponent10 == 0");
  static_assert(cuda::std::numeric_limits<A>::has_infinity == false,
                "cuda::std::numeric_limits<A>::has_infinity == false");
  static_assert(cuda::std::numeric_limits<A>::has_quiet_NaN == false,
                "cuda::std::numeric_limits<A>::has_quiet_NaN == false");
  static_assert(cuda::std::numeric_limits<A>::has_signaling_NaN == false,
                "cuda::std::numeric_limits<A>::has_signaling_NaN == false");
  static_assert(cuda::std::numeric_limits<A>::has_denorm == cuda::std::denorm_absent,
                "cuda::std::numeric_limits<A>::has_denorm == cuda::std::denorm_absent");
  static_assert(cuda::std::numeric_limits<A>::has_denorm_loss == false,
                "cuda::std::numeric_limits<A>::has_denorm_loss == false");
  assert(cuda::std::numeric_limits<A>::infinity() == A());
  assert(cuda::std::numeric_limits<A>::quiet_NaN() == A());
  assert(cuda::std::numeric_limits<A>::signaling_NaN() == A());
  assert(cuda::std::numeric_limits<A>::denorm_min() == A());
  static_assert(cuda::std::numeric_limits<A>::is_iec559 == false, "cuda::std::numeric_limits<A>::is_iec559 == false");
  static_assert(cuda::std::numeric_limits<A>::is_bounded == false, "cuda::std::numeric_limits<A>::is_bounded == false");
  static_assert(cuda::std::numeric_limits<A>::is_modulo == false, "cuda::std::numeric_limits<A>::is_modulo == false");
  static_assert(cuda::std::numeric_limits<A>::traps == false, "cuda::std::numeric_limits<A>::traps == false");
  static_assert(cuda::std::numeric_limits<A>::tinyness_before == false,
                "cuda::std::numeric_limits<A>::tinyness_before == false");
  static_assert(cuda::std::numeric_limits<A>::round_style == cuda::std::round_toward_zero,
                "cuda::std::numeric_limits<A>::round_style == cuda::std::round_toward_zero");

  return 0;
}
