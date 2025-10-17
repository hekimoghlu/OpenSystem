/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

// float_round_style

#include <uscl/std/limits>

#include "test_macros.h"

typedef char one;
struct two
{
  one _[2];
};

__host__ __device__ one test(cuda::std::float_round_style);
__host__ __device__ two test(int);

int main(int, char**)
{
  static_assert(cuda::std::round_indeterminate == -1, "cuda::std::round_indeterminate == -1");
  static_assert(cuda::std::round_toward_zero == 0, "cuda::std::round_toward_zero == 0");
  static_assert(cuda::std::round_to_nearest == 1, "cuda::std::round_to_nearest == 1");
  static_assert(cuda::std::round_toward_infinity == 2, "cuda::std::round_toward_infinity == 2");
  static_assert(cuda::std::round_toward_neg_infinity == 3, "cuda::std::round_toward_neg_infinity == 3");
  static_assert(sizeof(test(cuda::std::round_to_nearest)) == 1, "sizeof(test(cuda::std::round_to_nearest)) == 1");
  static_assert(sizeof(test(1)) == 2, "sizeof(test(1)) == 2");

  return 0;
}
