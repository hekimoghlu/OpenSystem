/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 12, 2025.
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

#ifndef TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
#define TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H

#include <uscl/std/cassert>
#include <uscl/std/iterator>

#include "test_macros.h"

struct PlusOne
{
  __host__ __device__ constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};

struct PlusOneMutable
{
  __host__ __device__ constexpr int operator()(int x) noexcept
  {
    return x + 1;
  }
};

struct PlusOneMayThrow
{
  __host__ __device__ constexpr int operator()(int x)
  {
    return x + 1;
  }
};

#if !TEST_COMPILER(NVRTC)
struct PlusOneHost
{
  constexpr PlusOneHost() noexcept {}
  constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // !TEST_COMPILER(NVRTC)

#if TEST_HAS_CUDA_COMPILER()
struct PlusOneDevice
{
  __device__ constexpr PlusOneDevice() noexcept {}
  __device__ constexpr int operator()(int x) const noexcept
  {
    return x + 1;
  }
};
#endif // TEST_HAS_CUDA_COMPILER()

struct NotDefaultConstructiblePlusOne
{
  __host__ __device__ constexpr NotDefaultConstructiblePlusOne(int) noexcept {}
  __host__ __device__ constexpr int operator()(int x) const
  {
    return x + 1;
  }
};

#endif // TEST_CUDA_ITERATOR_COUNTING_ITERATOR_H
