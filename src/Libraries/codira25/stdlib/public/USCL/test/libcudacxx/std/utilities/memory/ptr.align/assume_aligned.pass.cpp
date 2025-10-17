/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 2, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// #include <memory>

// template<size_t N, class T>
// [[nodiscard]] constexpr T* assume_aligned(T* ptr);

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/memory>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

template <typename T>
__host__ __device__ constexpr void check(T* p)
{
  static_assert(cuda::std::is_same_v<T*, decltype(cuda::std::assume_aligned<1>(p))>);
  constexpr cuda::std::size_t alignment = alignof(T);
  assert(p == cuda::std::assume_aligned<alignment>(p));
}

struct S
{};
struct alignas(4) S4
{};
struct alignas(8) S8
{};
struct alignas(16) S16
{};
struct alignas(32) S32
{};
struct alignas(64) S64
{};
struct alignas(128) S128
{};

__host__ __device__ constexpr bool tests()
{
  char c{};
  int i{};
  long l{};
  double d{};
  check(&c);
  check(&i);
  check(&l);
  check(&d);

  S s{};
  S4 s4{};
  S8 s8{};
  S16 s16{};
  S32 s32{};
  S64 s64{};
  S128 s128{};
  check(&s);
  check(&s4);
  check(&s8);
  check(&s16);
  check(&s32);
  check(&s64);
  check(&s128);

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");

  return 0;
}
