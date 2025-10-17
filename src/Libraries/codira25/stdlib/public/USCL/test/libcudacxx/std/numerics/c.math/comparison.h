/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

#ifndef TEST_CMATH_COMPARISON_H
#define TEST_CMATH_COMPARISON_H

#include <uscl/std/type_traits>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool eq(T lhs, T rhs) noexcept
{
  return lhs == rhs;
}

template <typename T, typename U, cuda::std::enable_if_t<cuda::std::is_arithmetic_v<U>, int> = 0>
__host__ __device__ constexpr bool eq(T lhs, U rhs) noexcept
{
  return eq(lhs, T(rhs));
}

#if _CCCL_HAS_NVFP16()
__host__ __device__ bool eq(__half lhs, __half rhs) noexcept
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return ::__heq(lhs, rhs);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
  return ::__half2float(lhs) == ::__half2float(rhs);
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
}
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
__host__ __device__ bool eq(__nv_bfloat16 lhs, __nv_bfloat16 rhs) noexcept
{
#  if _CCCL_CTK_AT_LEAST(12, 2)
  return ::__heq(lhs, rhs);
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 2) ^^^ / vvv !_CCCL_CTK_AT_LEAST(12, 2) vvv
  return ::__bfloat162float(lhs) == ::__bfloat162float(rhs);
#  endif // !_CCCL_CTK_AT_LEAST(12, 2)
}
#endif // _CCCL_HAS_NVBF16()

template <class Integer, cuda::std::enable_if_t<cuda::std::is_integral_v<Integer>, int> = 0>
__host__ __device__ bool is_about(Integer x, Integer y)
{
  return true;
}

__host__ __device__ bool is_about(float x, float y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-6);
}

__host__ __device__ bool is_about(double x, double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}

#if _CCCL_HAS_LONG_DOUBLE()
__host__ __device__ bool is_about(long double x, long double y)
{
  return (cuda::std::abs((x - y) / (x + y)) < 1.e-14);
}
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
__host__ __device__ bool is_about(__half x, __half y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __half(1e-3));
}
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
__host__ __device__ bool is_about(__nv_bfloat16 x, __nv_bfloat16 y)
{
  return (cuda::std::fabs((x - y) / (x + y)) <= __nv_bfloat16(5e-3));
}
#endif // _LIBCUDACXX_HAS_NVBF16()

#endif // TEST_CMATH_COMPARISON_H
