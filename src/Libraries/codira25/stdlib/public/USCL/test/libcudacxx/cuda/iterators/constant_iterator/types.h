/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 11, 2024.
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

#ifndef TEST_CUDA_ITERATOR_CONSTANT_ITERATOR_H
#define TEST_CUDA_ITERATOR_CONSTANT_ITERATOR_H

#include <uscl/std/iterator>

#include "test_macros.h"

struct DefaultConstructibleTo42
{
  int val_;

  __host__ __device__ constexpr DefaultConstructibleTo42(const int val = 42) noexcept
      : val_(val)
  {}

  __host__ __device__ friend constexpr bool
  operator==(DefaultConstructibleTo42 lhs, DefaultConstructibleTo42 rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend constexpr bool
  operator!=(DefaultConstructibleTo42 lhs, DefaultConstructibleTo42 rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
};

struct NotDefaultConstructible
{
  int val_;

  __host__ __device__ constexpr NotDefaultConstructible(const int val) noexcept
      : val_(val)
  {}
  __host__ __device__ friend constexpr bool operator==(NotDefaultConstructible lhs, NotDefaultConstructible rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
  __host__ __device__ friend constexpr bool operator!=(NotDefaultConstructible lhs, NotDefaultConstructible rhs) noexcept
  {
    return lhs.val_ == rhs.val_;
  }
};

#endif // TEST_CUDA_ITERATOR_CONSTANT_ITERATOR_H
