/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#ifndef LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_
#define LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_

#include <uscl/std/concepts>

#include "test_macros.h"

#if TEST_STD_VER > 2017
// This overload should never be called. It exists solely to force subsumption.
template <cuda::std::integral I>
__host__ __device__ constexpr bool CheckSubsumption(I)
{
  return false;
}

template <cuda::std::integral I>
  requires cuda::std::signed_integral<I> && (!cuda::std::unsigned_integral<I>)
__host__ __device__ constexpr bool CheckSubsumption(I)
{
  return cuda::std::is_signed_v<I>;
}

template <cuda::std::integral I>
  requires cuda::std::unsigned_integral<I> && (!cuda::std::signed_integral<I>)
__host__ __device__ constexpr bool CheckSubsumption(I)
{
  return cuda::std::is_unsigned_v<I>;
}
#endif // TEST_STD_VER > 2017

enum ClassicEnum
{
  a,
  b,
  c
};
enum class ScopedEnum
{
  x,
  y,
  z
};
struct EmptyStruct
{};

#endif // LIBCXX_TEST_CONCEPTS_LANG_CONCEPTS_ARITHMETIC_H_
