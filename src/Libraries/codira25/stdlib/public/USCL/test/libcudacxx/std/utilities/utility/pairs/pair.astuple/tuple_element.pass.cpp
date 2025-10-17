/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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

// <utility>

// template <class T1, class T2> struct pair

// tuple_element<I, pair<T1, T2> >::type

#include <uscl/std/utility>

#include "test_macros.h"

template <class T1, class T2>
__host__ __device__ void test()
{
  {
    typedef T1 Exp1;
    typedef T2 Exp2;
    typedef cuda::std::pair<T1, T2> P;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
  {
    typedef T1 const Exp1;
    typedef T2 const Exp2;
    typedef cuda::std::pair<T1, T2> const P;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
  {
    typedef T1 volatile Exp1;
    typedef T2 volatile Exp2;
    typedef cuda::std::pair<T1, T2> volatile P;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
  {
    typedef T1 const volatile Exp1;
    typedef T2 const volatile Exp2;
    typedef cuda::std::pair<T1, T2> const volatile P;
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<0, P>::type, Exp1>::value), "");
    static_assert((cuda::std::is_same<typename cuda::std::tuple_element<1, P>::type, Exp2>::value), "");
  }
}

int main(int, char**)
{
  test<int, short>();
  test<int*, char>();

  return 0;
}
