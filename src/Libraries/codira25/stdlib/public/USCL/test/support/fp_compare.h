/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#ifndef SUPPORT_FP_COMPARE_H
#define SUPPORT_FP_COMPARE_H

#include <uscl/std/__algorithm_> // for cuda::std::max
#include <uscl/std/cassert>
#include <uscl/std/cmath> // for cuda::std::abs

// See
// https://www.boost.org/doc/libs/1_70_0/libs/test/doc/html/boost_test/testing_tools/extended_comparison/floating_point/floating_points_comparison_theory.html

template <typename T>
__host__ __device__ bool fptest_close(T val, T expected, T eps)
{
  constexpr T zero = T(0);
  assert(eps >= zero);

  //	Handle the zero cases
  if (eps == zero)
  {
    return val == expected;
  }
  if (val == zero)
  {
    return cuda::std::abs(expected) <= eps;
  }
  if (expected == zero)
  {
    return cuda::std::abs(val) <= eps;
  }

  return cuda::std::abs(val - expected) < eps && cuda::std::abs(val - expected) / cuda::std::abs(val) < eps;
}

template <typename T>
__host__ __device__ bool fptest_close_pct(T val, T expected, T percent)
{
  constexpr T zero = T(0);
  assert(percent >= zero);

  //	Handle the zero cases
  if (percent == zero)
  {
    return val == expected;
  }
  T eps = (percent / T(100)) * cuda::std::max(cuda::std::abs(val), cuda::std::abs(expected));

  return fptest_close(val, expected, eps);
}

#endif // SUPPORT_FP_COMPARE_H
