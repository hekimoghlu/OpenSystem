/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 8, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_ITERATORS_ITERATOR_PRIMITIVES_RANGE_ITER_OPS_TYPES_H
#define TEST_STD_ITERATORS_ITERATOR_PRIMITIVES_RANGE_ITER_OPS_TYPES_H

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/iterator>
#include <uscl/std/utility>

#include "test_iterators.h" // for the fallthrough base() function

class distance_apriori_sentinel
{
public:
  distance_apriori_sentinel() = default;
  __host__ __device__ constexpr explicit distance_apriori_sentinel(cuda::std::ptrdiff_t const count)
      : count_(count)
  {}

  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ friend constexpr bool operator==(distance_apriori_sentinel const, It const&)
  {
    assert(false && "difference op should take precedence");
    return false;
  }

#if TEST_STD_VER < 2020
  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ friend constexpr bool operator==(It const&, distance_apriori_sentinel const)
  {
    assert(false && "difference op should take precedence");
    return false;
  }

  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ friend constexpr bool operator!=(distance_apriori_sentinel const, It const&)
  {
    assert(false && "difference op should take precedence");
    return true;
  }

  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ friend constexpr bool operator!=(It const&, distance_apriori_sentinel const)
  {
    assert(false && "difference op should take precedence");
    return true;
  }
#endif

  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(It const&, distance_apriori_sentinel const y)
  {
    return -y.count_;
  }

  template <class It, cuda::std::enable_if_t<cuda::std::input_or_output_iterator<It>, int> = 0>
  __host__ __device__ friend constexpr cuda::std::ptrdiff_t operator-(distance_apriori_sentinel const x, It const&)
  {
    return x.count_;
  }

private:
  cuda::std::ptrdiff_t count_ = 0;
};

// Sentinel type that can be assigned to an iterator. This is to test the cases where the
// various iterator operations use assignment instead of successive increments/decrements.
template <class It>
class assignable_sentinel
{
public:
  explicit assignable_sentinel() = default;
  __host__ __device__ constexpr explicit assignable_sentinel(const It& it)
      : base_(base(it))
  {}
  __host__ __device__ constexpr operator It() const
  {
    return It(base_);
  }
  __host__ __device__ friend constexpr bool operator==(const assignable_sentinel& s, const It& other)
  {
    return s.base_ == base(other);
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator==(const It& other, const assignable_sentinel& s)
  {
    return s.base_ == base(other);
  }
  __host__ __device__ friend constexpr bool operator!=(const assignable_sentinel& s, const It& other)
  {
    return s.base_ != base(other);
  }
  __host__ __device__ friend constexpr bool operator!=(const It& other, const assignable_sentinel& s)
  {
    return s.base_ != base(other);
  }
#endif
  __host__ __device__ friend constexpr It base(const assignable_sentinel& s)
  {
    return It(s.base_);
  }

private:
  decltype(base(cuda::std::declval<It>())) base_;
};

template <class It>
__host__ __device__ assignable_sentinel(const It&) -> assignable_sentinel<It>;

#endif // TEST_STD_ITERATORS_ITERATOR_PRIMITIVES_RANGE_ITER_OPS_TYPES_H
