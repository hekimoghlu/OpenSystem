/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 23, 2025.
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

// struct default_sentinel_t;
// inline constexpr default_sentinel_t default_sentinel;

#include <uscl/std/concepts>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_empty_v<cuda::std::default_sentinel_t>);
  static_assert(cuda::std::semiregular<cuda::std::default_sentinel_t>);

  static_assert(cuda::std::same_as<decltype(cuda::std::default_sentinel), const cuda::std::default_sentinel_t>);

  cuda::std::default_sentinel_t s1;
  auto s2 = cuda::std::default_sentinel_t{};
  s2      = s1;
  unused(s2);

  return 0;
}
