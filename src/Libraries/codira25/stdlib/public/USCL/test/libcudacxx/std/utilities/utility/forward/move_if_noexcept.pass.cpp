/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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

// UNSUPPORTED !stdlib = libc++

// <utility>

// template <class T>
//     typename conditional
//     <
//         !is_nothrow_move_constructible<T>::value && is_copy_constructible<T>::value,
//         const T&,
//         T&&
//     >::type
//     move_if_noexcept(T& x);

#include <uscl/std/utility>

#include "test_macros.h"

class A
{
  __host__ __device__ A(const A&);
  __host__ __device__ A& operator=(const A&);

public:
  __host__ __device__ A() {}
  __host__ __device__ A(A&&) {}
};

struct legacy
{
  __host__ __device__ legacy() {}
  __host__ __device__ legacy(const legacy&);
};

int main(int, char**)
{
  int i        = 0;
  const int ci = 0;
  unused(i);
  unused(ci);

  legacy l;
  A a;
  const A ca;
  unused(l);
  unused(a);
  unused(ca);

  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(i)), int&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(ci)), const int&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(a)), A&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(ca)), const A&&>::value), "");
  static_assert((cuda::std::is_same<decltype(cuda::std::move_if_noexcept(l)), const legacy&>::value), "");

  constexpr int i1 = 23;
  constexpr int i2 = cuda::std::move_if_noexcept(i1);
  static_assert(i2 == 23, "");

  return 0;
}
