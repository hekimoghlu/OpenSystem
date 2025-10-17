/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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
// struct unwrap_ref_decay;
//
// template <class T>
// using unwrap_ref_decay_t = typename unwrap_ref_decay<T>::type;

#include <uscl/std/functional>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

template <typename T, typename Result>
__host__ __device__ void check()
{
  static_assert(cuda::std::is_same_v<typename cuda::std::unwrap_ref_decay<T>::type, Result>);
  static_assert(cuda::std::is_same_v<typename cuda::std::unwrap_ref_decay<T>::type, cuda::std::unwrap_ref_decay_t<T>>);
}

struct T
{};

int main(int, char**)
{
  check<T, T>();
  check<T&, T>();
  check<T const, T>();
  check<T const&, T>();
  check<T*, T*>();
  check<T const*, T const*>();
  check<T[3], T*>();
  check<T const[3], T const*>();
  check<T(), T (*)()>();
  check<T(int) const, T(int) const>();
  check<T(int) &, T(int) &>();
  check<T(int) &&, T(int) &&>();

  check<cuda::std::reference_wrapper<T>, T&>();
  check<cuda::std::reference_wrapper<T>&, T&>();
  check<cuda::std::reference_wrapper<T const>, T const&>();
  check<cuda::std::reference_wrapper<T const>&, T const&>();
  check<cuda::std::reference_wrapper<T*>, T*&>();
  check<cuda::std::reference_wrapper<T*>&, T*&>();
  check<cuda::std::reference_wrapper<T const*>, T const*&>();
  check<cuda::std::reference_wrapper<T const*>&, T const*&>();
  check<cuda::std::reference_wrapper<T[3]>, T(&)[3]>();
  check<cuda::std::reference_wrapper<T[3]>&, T(&)[3]>();
  check<cuda::std::reference_wrapper<T()>, T (&)()>();
  check<cuda::std::reference_wrapper<T()>&, T (&)()>();

  return 0;
}
