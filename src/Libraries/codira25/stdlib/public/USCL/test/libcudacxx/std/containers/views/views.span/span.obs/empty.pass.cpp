/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

// <span>

// [[nodiscard]] constexpr bool empty() const noexcept;
//

#include <uscl/std/cassert>
#include <uscl/std/span>

#include "test_macros.h"

struct A
{};
constexpr int iArr1[]            = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
TEST_GLOBAL_VARIABLE int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{
  static_assert(noexcept(cuda::std::span<int>().empty()), "");
  static_assert(noexcept(cuda::std::span<int, 0>().empty()), "");

  static_assert(cuda::std::span<int>().empty(), "");
  static_assert(cuda::std::span<long>().empty(), "");
  static_assert(cuda::std::span<double>().empty(), "");
  static_assert(cuda::std::span<A>().empty(), "");

  static_assert(cuda::std::span<int, 0>().empty(), "");
  static_assert(cuda::std::span<long, 0>().empty(), "");
  static_assert(cuda::std::span<double, 0>().empty(), "");
  static_assert(cuda::std::span<A, 0>().empty(), "");

  static_assert(!cuda::std::span<const int>(iArr1, 1).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 2).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 3).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 4).empty(), "");
  static_assert(!cuda::std::span<const int>(iArr1, 5).empty(), "");

  assert((cuda::std::span<int>().empty()));
  assert((cuda::std::span<long>().empty()));
  assert((cuda::std::span<double>().empty()));
  assert((cuda::std::span<A>().empty()));

  assert((cuda::std::span<int, 0>().empty()));
  assert((cuda::std::span<long, 0>().empty()));
  assert((cuda::std::span<double, 0>().empty()));
  assert((cuda::std::span<A, 0>().empty()));

  assert(!(cuda::std::span<int, 1>(iArr2, 1).empty()));
  assert(!(cuda::std::span<int, 2>(iArr2, 2).empty()));
  assert(!(cuda::std::span<int, 3>(iArr2, 3).empty()));
  assert(!(cuda::std::span<int, 4>(iArr2, 4).empty()));
  assert(!(cuda::std::span<int, 5>(iArr2, 5).empty()));

  return 0;
}
