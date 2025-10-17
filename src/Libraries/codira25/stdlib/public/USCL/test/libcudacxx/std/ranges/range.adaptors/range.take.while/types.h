/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_WHILE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_WHILE_TYPES_H

#include <uscl/std/array>
#include <uscl/std/functional>
#include <uscl/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

template <class T>
struct BufferViewBase : cuda::std::ranges::view_base
{
  T* buffer_;
  cuda::std::size_t size_;

  template <cuda::std::size_t N>
  __host__ __device__ constexpr BufferViewBase(T (&b)[N])
      : buffer_(b)
      , size_(N)
  {}

  template <cuda::std::size_t N>
  __host__ __device__ constexpr BufferViewBase(cuda::std::array<T, N>& arr)
      : buffer_(arr.data())
      , size_(N)
  {}
};

using IntBufferViewBase = BufferViewBase<int>;

struct SimpleView : IntBufferViewBase
{
#if TEST_COMPILER(NVRTC) // nvbug 3961621
  SimpleView() = default;

  template <class T>
  __host__ __device__ constexpr SimpleView(T&& input)
      : IntBufferViewBase(cuda::std::forward<T>(input))
  {}
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  // ^^^ TEST_COMPILER(NVRTC) ^^^ / vvv !TEST_COMPILER(NVRTC) vvv
  using IntBufferViewBase::IntBufferViewBase;
#endif // !TEST_COMPILER(NVRTC)
  __host__ __device__ constexpr int* begin() const
  {
    return buffer_;
  }
  __host__ __device__ constexpr int* end() const
  {
    return buffer_ + size_;
  }
};
static_assert(cuda::std::ranges::__simple_view<SimpleView>);

struct ConstNotRange : IntBufferViewBase
{
#if TEST_COMPILER(NVRTC) // nvbug 3961621
  ConstNotRange() = default;

  template <class T>
  __host__ __device__ constexpr ConstNotRange(T&& input)
      : IntBufferViewBase(cuda::std::forward<T>(input))
  {}
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  // ^^^ TEST_COMPILER(NVRTC) ^^^ / vvv !TEST_COMPILER(NVRTC) vvv
  using IntBufferViewBase::IntBufferViewBase;
#endif // !TEST_COMPILER(NVRTC)
  __host__ __device__ constexpr int* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr int* end()
  {
    return buffer_ + size_;
  }
};
static_assert(cuda::std::ranges::view<ConstNotRange>);
static_assert(!cuda::std::ranges::range<const ConstNotRange>);

struct NonSimple : IntBufferViewBase
{
#if TEST_COMPILER(NVRTC) // nvbug 3961621
  NonSimple() = default;

  template <class T>
  __host__ __device__ constexpr NonSimple(T&& input)
      : IntBufferViewBase(cuda::std::forward<T>(input))
  {}
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  // ^^^ TEST_COMPILER(NVRTC) ^^^ / vvv !TEST_COMPILER(NVRTC) vvv
  using IntBufferViewBase::IntBufferViewBase;
#endif // !TEST_COMPILER(NVRTC)
  __host__ __device__ constexpr const int* begin() const
  {
    return buffer_;
  }
  __host__ __device__ constexpr const int* end() const
  {
    return buffer_ + size_;
  }
  __host__ __device__ constexpr int* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr int* end()
  {
    return buffer_ + size_;
  }
};
static_assert(cuda::std::ranges::view<NonSimple>);
static_assert(!cuda::std::ranges::__simple_view<NonSimple>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_WHILE_TYPES_H
