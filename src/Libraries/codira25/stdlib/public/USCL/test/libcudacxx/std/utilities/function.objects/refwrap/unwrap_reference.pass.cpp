/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 8, 2023.
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
//
// template <class T>
// struct unwrap_reference;
//
// template <class T>
// using unwrap_reference_t = typename unwrap_reference<T>::type;

#include <uscl/std/functional>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

template <typename T, typename Expected>
__host__ __device__ void check_equal()
{
  static_assert(cuda::std::is_same_v<typename cuda::std::unwrap_reference<T>::type, Expected>);
  static_assert(cuda::std::is_same_v<typename cuda::std::unwrap_reference<T>::type, cuda::std::unwrap_reference_t<T>>);
}

template <typename T>
__host__ __device__ void check()
{
  check_equal<T, T>();
  check_equal<T&, T&>();
  check_equal<T const, T const>();
  check_equal<T const&, T const&>();

  check_equal<cuda::std::reference_wrapper<T>, T&>();
  check_equal<cuda::std::reference_wrapper<T const>, T const&>();
}

struct T
{};

int main(int, char**)
{
  check<T>();
  check<int>();
  check<float>();

  check<T*>();
  check<int*>();
  check<float*>();

  return 0;
}
