/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 1, 2025.
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

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_CONVERTIBLE_TO_INTEGRAL_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_CONVERTIBLE_TO_INTEGRAL_H

#include "CommonHelpers.h"
#include "test_macros.h"

struct IntType
{
  int val;
  IntType() = default;
  __host__ __device__ constexpr IntType(int v) noexcept
      : val(v){};

  __host__ __device__ constexpr bool operator==(const IntType& rhs) const
  {
    return val == rhs.val;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ constexpr bool operator!=(const IntType& rhs) const
  {
    return val != rhs.val;
  }
#endif // TEST_STD_VER < 2020
  __host__ __device__ constexpr operator int() const noexcept
  {
    return val;
  }
  __host__ __device__ constexpr operator unsigned char() const
  {
    return static_cast<unsigned char>(val);
  }
  __host__ __device__ constexpr operator char() const noexcept
  {
    return static_cast<char>(val);
  }
};

// only non-const convertible
struct IntTypeNC
{
  int val;
  IntTypeNC() = default;
  __host__ __device__ constexpr IntTypeNC(int v) noexcept
      : val(v){};

  __host__ __device__ constexpr bool operator==(const IntType& rhs) const
  {
    return val == rhs.val;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ constexpr bool operator!=(const IntType& rhs) const
  {
    return val != rhs.val;
  }
#endif // TEST_STD_VER < 2020
  __host__ __device__ constexpr operator int() noexcept
  {
    return val;
  }
  __host__ __device__ constexpr operator unsigned()
  {
    return static_cast<unsigned>(val);
  }
  __host__ __device__ constexpr operator char() noexcept
  {
    return static_cast<char>(val);
  }
};

// weird configurability of convertibility to int
template <bool conv_c, bool conv_nc, bool ctor_nt_c, bool ctor_nt_nc>
struct IntConfig
{
  int val;
  __host__ __device__ constexpr explicit IntConfig(int val_)
      : val(val_)
  {}
  template <bool Convertible = conv_nc, cuda::std::enable_if_t<Convertible, int> = 0>
  __host__ __device__ constexpr operator int() noexcept(ctor_nt_nc)
  {
    return val;
  }
  template <bool Convertible = conv_c, cuda::std::enable_if_t<Convertible, int> = 0>
  __host__ __device__ constexpr operator int() const noexcept(ctor_nt_c)
  {
    return val;
  }
};

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_CONVERTIBLE_TO_INTEGRAL_H
