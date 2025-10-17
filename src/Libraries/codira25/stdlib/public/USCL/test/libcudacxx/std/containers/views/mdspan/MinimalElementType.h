/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H

#include <uscl/std/utility>

#include "CommonHelpers.h"
#include "test_macros.h"

// Idiosyncratic element type for mdspan
// Make sure we don't assume copyable, default constructible, movable etc.
struct MinimalElementType
{
  template <class T, size_t N>
  friend struct ElementPool;

  struct tag
  {
    __host__ __device__ constexpr tag(int) noexcept {}

    __host__ __device__ constexpr operator int() noexcept
    {
      return 42;
    }
  };

  int val;
  constexpr MinimalElementType()                                     = delete;
  constexpr MinimalElementType& operator=(const MinimalElementType&) = delete;
  __host__ __device__ constexpr explicit MinimalElementType(int v) noexcept
      : val(v)
  {}

  __host__ __device__ constexpr MinimalElementType(tag) noexcept
      : val(42)
  {}

  // MSVC cannot list init the element and complains about the deleted copy constructor. Emulate via private
#if _CCCL_COMPILER(MSVC)

private:
  constexpr MinimalElementType(const MinimalElementType&) = default;
#else // ^^^ _CCCL_COMPILER(MSVC2019) ^^^ / vvv !_CCCL_COMPILER(MSVC2019) vvv
  constexpr MinimalElementType(const MinimalElementType&) = delete;
#endif // !_CCCL_COMPILER(MSVC2019)
};

// Helper class to create pointer to MinimalElementType
template <class T, size_t N>
struct ElementPool
{
private:
  template <int... Indices>
  __host__ __device__ constexpr ElementPool(cuda::std::integer_sequence<int, Indices...>)
      : ptr_{T{MinimalElementType::tag{Indices}}...}
  {}

public:
  __host__ __device__ constexpr ElementPool()
      : ElementPool(cuda::std::make_integer_sequence<int, N>())
  {}

  __host__ __device__ constexpr T* get_ptr()
  {
    return ptr_;
  }

private:
  T ptr_[N];
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_MINIMAL_ELEMENT_TYPE_H
