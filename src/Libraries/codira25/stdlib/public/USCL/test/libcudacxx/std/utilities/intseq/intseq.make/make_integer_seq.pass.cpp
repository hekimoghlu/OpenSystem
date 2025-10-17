/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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

// <utility>

// template<class T, T N>
//   using make_integer_sequence = integer_sequence<T, 0, 1, ..., N-1>;

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<int, 0>, cuda::std::integer_sequence<int>>::value,
                "");
  static_assert(
    cuda::std::is_same<cuda::std::make_integer_sequence<int, 1>, cuda::std::integer_sequence<int, 0>>::value, "");
  static_assert(
    cuda::std::is_same<cuda::std::make_integer_sequence<int, 2>, cuda::std::integer_sequence<int, 0, 1>>::value, "");
  static_assert(
    cuda::std::is_same<cuda::std::make_integer_sequence<int, 3>, cuda::std::integer_sequence<int, 0, 1, 2>>::value, "");

  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 0>,
                                   cuda::std::integer_sequence<unsigned long long>>::value,
                "");
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 1>,
                                   cuda::std::integer_sequence<unsigned long long, 0>>::value,
                "");
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 2>,
                                   cuda::std::integer_sequence<unsigned long long, 0, 1>>::value,
                "");
  static_assert(cuda::std::is_same<cuda::std::make_integer_sequence<unsigned long long, 3>,
                                   cuda::std::integer_sequence<unsigned long long, 0, 1, 2>>::value,
                "");

  return 0;
}
