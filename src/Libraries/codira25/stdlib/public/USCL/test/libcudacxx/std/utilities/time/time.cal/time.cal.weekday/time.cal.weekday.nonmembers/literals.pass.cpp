/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 4, 2022.
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
//
//===----------------------------------------------------------------------===//

// <chrono>

// inline constexpr weekday   Sunday{0};
// inline constexpr weekday   Monday{1};
// inline constexpr weekday   Tuesday{2};
// inline constexpr weekday   Wednesday{3};
// inline constexpr weekday   Thursday{4};
// inline constexpr weekday   Friday{5};
// inline constexpr weekday   Saturday{6};

#include <uscl/std/cassert>
#include <uscl/std/chrono>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::weekday, decltype(cuda::std::chrono::Sunday)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::weekday, decltype(cuda::std::chrono::Monday)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::weekday, decltype(cuda::std::chrono::Tuesday)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::weekday, decltype(cuda::std::chrono::Wednesday)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::weekday, decltype(cuda::std::chrono::Thursday)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::weekday, decltype(cuda::std::chrono::Friday)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::weekday, decltype(cuda::std::chrono::Saturday)>);

  static_assert(cuda::std::chrono::Sunday == cuda::std::chrono::weekday(0), "");
  static_assert(cuda::std::chrono::Monday == cuda::std::chrono::weekday(1), "");
  static_assert(cuda::std::chrono::Tuesday == cuda::std::chrono::weekday(2), "");
  static_assert(cuda::std::chrono::Wednesday == cuda::std::chrono::weekday(3), "");
  static_assert(cuda::std::chrono::Thursday == cuda::std::chrono::weekday(4), "");
  static_assert(cuda::std::chrono::Friday == cuda::std::chrono::weekday(5), "");
  static_assert(cuda::std::chrono::Saturday == cuda::std::chrono::weekday(6), "");

  assert(cuda::std::chrono::Sunday == cuda::std::chrono::weekday(0));
  assert(cuda::std::chrono::Monday == cuda::std::chrono::weekday(1));
  assert(cuda::std::chrono::Tuesday == cuda::std::chrono::weekday(2));
  assert(cuda::std::chrono::Wednesday == cuda::std::chrono::weekday(3));
  assert(cuda::std::chrono::Thursday == cuda::std::chrono::weekday(4));
  assert(cuda::std::chrono::Friday == cuda::std::chrono::weekday(5));
  assert(cuda::std::chrono::Saturday == cuda::std::chrono::weekday(6));

  assert(cuda::std::chrono::Sunday.c_encoding() == 0);
  assert(cuda::std::chrono::Monday.c_encoding() == 1);
  assert(cuda::std::chrono::Tuesday.c_encoding() == 2);
  assert(cuda::std::chrono::Wednesday.c_encoding() == 3);
  assert(cuda::std::chrono::Thursday.c_encoding() == 4);
  assert(cuda::std::chrono::Friday.c_encoding() == 5);
  assert(cuda::std::chrono::Saturday.c_encoding() == 6);

  return 0;
}
