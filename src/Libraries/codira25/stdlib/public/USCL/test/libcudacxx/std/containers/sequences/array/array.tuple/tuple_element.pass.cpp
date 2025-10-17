/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// tuple_element<I, array<T, N> >::type

#include <uscl/std/array>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test()
{
  {
    typedef T Exp;
    typedef cuda::std::array<T, 3> C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
  {
    typedef T const Exp;
    typedef cuda::std::array<T, 3> const C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
  {
    typedef T volatile Exp;
    typedef cuda::std::array<T, 3> volatile C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
  {
    typedef T const volatile Exp;
    typedef cuda::std::array<T, 3> const volatile C;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, C>::type, Exp>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<2, C>::type, Exp>::value), "");
  }
}

int main(int, char**)
{
  test<double>();
  test<int>();

  return 0;
}
