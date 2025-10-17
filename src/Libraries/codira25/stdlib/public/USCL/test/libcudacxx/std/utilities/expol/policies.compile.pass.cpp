/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// class sequenced_policy;
// class parallel_policy;
// class parallel_unsequenced_policy;
// class unsequenced_policy; // since C++20
//
// inline constexpr sequenced_policy seq = implementation-defined;
// inline constexpr parallel_policy par = implementation-defined;
// inline constexpr parallel_unsequenced_policy par_unseq = implementation-defined;
// inline constexpr unsequenced_policy unseq = implementation-defined; // since C++20

// UNSUPPORTED: libcpp-has-no-incomplete-pstl

#include <uscl/std/execution>
#include <uscl/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  namespace execution = cuda::std::execution;

  assert(execution::seq == execution::seq);
  assert(execution::par == execution::par);
  assert(execution::par_unseq == execution::par_unseq);
  assert(execution::unseq == execution::unseq);

  assert(!(execution::seq != execution::seq));
  assert(!(execution::par != execution::par));
  assert(!(execution::par_unseq != execution::par_unseq));
  assert(!(execution::unseq != execution::unseq));

  assert(!(execution::seq == execution::unseq));
  assert(!(execution::par == execution::seq));
  assert(!(execution::par_unseq == execution::par));
  assert(!(execution::unseq == execution::par_unseq));

  assert(execution::seq != execution::unseq);
  assert(execution::par != execution::seq);
  assert(execution::par_unseq != execution::par);
  assert(execution::unseq != execution::par_unseq);

  return true;
}

template <class T, class Policy>
inline constexpr bool is_same_v = cuda::std::is_same_v<cuda::std::remove_cvref_t<T>, Policy>;

static_assert(is_same_v<decltype(cuda::std::execution::seq), cuda::std::execution::sequenced_policy>);
static_assert(is_same_v<decltype(cuda::std::execution::par), cuda::std::execution::parallel_policy>);
static_assert(is_same_v<decltype(cuda::std::execution::par_unseq), cuda::std::execution::parallel_unsequenced_policy>);
static_assert(is_same_v<decltype(cuda::std::execution::unseq), cuda::std::execution::unsequenced_policy>);

int main(int, char**)
{
  static_assert(test());

  return 0;
}
