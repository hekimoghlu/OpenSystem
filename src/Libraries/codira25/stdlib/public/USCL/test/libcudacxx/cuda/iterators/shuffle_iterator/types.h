/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 1, 2023.
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

#ifndef TEST_CUDA_ITERATOR_SHUFFLE_ITERATOR_H
#define TEST_CUDA_ITERATOR_SHUFFLE_ITERATOR_H

#include <uscl/std/cstdint>

#include "test_macros.h"

struct fake_rng
{
  using result_type = uint32_t;

  constexpr fake_rng() = default;

  [[nodiscard]] __host__ __device__ constexpr result_type operator()() noexcept
  {
    return __random_indices[__start++ % 5];
  }

  // Needed for uniform_int_distribution
  [[nodiscard]] __host__ __device__ static constexpr result_type min() noexcept
  {
    return 0;
  }

  [[nodiscard]] __host__ __device__ static constexpr result_type max() noexcept
  {
    return 5;
  }

  uint32_t __start{0};
  uint32_t __random_indices[5] = {4, 1, 2, 0, 3};
};
static_assert(cuda::std::__cccl_random_is_valid_urng<fake_rng>);

template <bool HasConstructor = true, bool HasNothrowCallOperator = true>
struct fake_bijection
{
  using index_type = uint32_t;

  constexpr fake_bijection() = default;

  _CCCL_TEMPLATE(class RNG, bool HasConstructor2 = HasConstructor)
  _CCCL_REQUIRES(HasConstructor2)
  __host__ __device__ constexpr fake_bijection(index_type, RNG&&) noexcept {}

  [[nodiscard]] __host__ __device__ constexpr index_type size() const noexcept(HasNothrowCallOperator)
  {
    return 5;
  }

  [[nodiscard]] __host__ __device__ constexpr index_type operator()(index_type n) const noexcept(HasNothrowCallOperator)
  {
    return __random_indices[n];
  }

  uint32_t __random_indices[5] = {4, 1, 2, 0, 3};
};

#endif // TEST_CUDA_ITERATOR_SHUFFLE_ITERATOR_H
