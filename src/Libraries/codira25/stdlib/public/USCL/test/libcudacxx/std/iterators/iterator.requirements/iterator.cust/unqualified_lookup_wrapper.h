/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#ifndef LIBCUDACXX_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER
#define LIBCUDACXX_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER

#include <uscl/std/iterator>
#include <uscl/std/utility>

#include "test_macros.h"

namespace check_unqualified_lookup
{
// Wrapper around an iterator for testing unqualified calls to `iter_move` and `iter_swap`.
template <typename I>
class unqualified_lookup_wrapper
{
public:
  unqualified_lookup_wrapper() = default;

  __host__ __device__ constexpr explicit unqualified_lookup_wrapper(I i) noexcept
      : base_(cuda::std::move(i))
  {}

  __host__ __device__ constexpr decltype(auto) operator*() const noexcept
  {
    return *base_;
  }

  __host__ __device__ constexpr unqualified_lookup_wrapper& operator++() noexcept
  {
    ++base_;
    return *this;
  }

  __host__ __device__ constexpr void operator++(int) noexcept
  {
    ++base_;
  }

  __host__ __device__ constexpr bool operator==(unqualified_lookup_wrapper const& other) const noexcept
  {
    return base_ == other.base_;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ constexpr bool operator!=(unqualified_lookup_wrapper const& other) const noexcept
  {
    return base_ != other.base_;
  }
#endif

  // Delegates `cuda::std::ranges::iter_move` for the underlying iterator. `noexcept(false)` will be used
  // to ensure that the unqualified-lookup overload is chosen.
  __host__ __device__ friend constexpr decltype(auto) iter_move(unqualified_lookup_wrapper& i) noexcept(false)
  {
    return cuda::std::ranges::iter_move(i.base_);
  }

private:
  I base_ = I{};
};

enum unscoped_enum
{
  a,
  b,
  c
};
__host__ __device__ constexpr unscoped_enum iter_move(unscoped_enum& e) noexcept(false)
{
  return e;
}

enum class scoped_enum
{
  a,
  b,
  c
};
__host__ __device__ constexpr scoped_enum* iter_move(scoped_enum&) noexcept
{
  return nullptr;
}

union some_union
{
  int x;
  double y;
};
__host__ __device__ constexpr int iter_move(some_union& u) noexcept(false)
{
  return u.x;
}

} // namespace check_unqualified_lookup

class move_tracker
{
public:
  move_tracker() = default;
  __host__ __device__ constexpr move_tracker(move_tracker&& other) noexcept
      : moves_{other.moves_ + 1}
  {
    other.moves_ = 0;
  }
  __host__ __device__ constexpr move_tracker& operator=(move_tracker&& other) noexcept
  {
    moves_       = other.moves_ + 1;
    other.moves_ = 0;
    return *this;
  }

  move_tracker(move_tracker const& other)            = delete;
  move_tracker& operator=(move_tracker const& other) = delete;

  __host__ __device__ constexpr int moves() const noexcept
  {
    return moves_;
  }

private:
  int moves_ = 0;
};

#endif // LIBCUDACXX_TEST_STD_ITERATOR_UNQUALIFIED_LOOKUP_WRAPPER
