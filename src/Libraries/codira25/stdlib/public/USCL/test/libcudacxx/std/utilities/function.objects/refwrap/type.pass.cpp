/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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

// <functional>

// reference_wrapper

// check for member typedef type

// #include <uscl/std/functional>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

class C
{};

int main(int, char**)
{
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<C>::type, C>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<void()>::type, void()>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<int*(double*)>::type, int*(double*)>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<void (*)()>::type, void (*)()>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<int* (*) (double*)>::type, int* (*) (double*)>::value),
                "");
  static_assert(
    (cuda::std::is_same<cuda::std::reference_wrapper<int* (C::*) (double*)>::type, int* (C::*) (double*)>::value), "");
  static_assert((cuda::std::is_same<cuda::std::reference_wrapper<int (C::*)(double*) const volatile>::type,
                                    int (C::*)(double*) const volatile>::value),
                "");

  return 0;
}
