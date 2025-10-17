/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 27, 2021.
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

// <cuda/std/chrono>

// template <class Rep1, class Period1, class Rep2, class Period2>
// struct common_type<chrono::duration<Rep1, Period1>, chrono::duration<Rep2, Period2>>
// {
//     typedef chrono::duration<typename common_type<Rep1, Rep2>::type, see below }> type;
// };

#include <uscl/std/chrono>

template <class D1, class D2, class De>
__host__ __device__ void test()
{
  typedef typename cuda::std::common_type<D1, D2>::type Dc;
  static_assert((cuda::std::is_same<Dc, De>::value), "");
}

int main(int, char**)
{
  test<cuda::std::chrono::duration<int, cuda::std::ratio<1, 100>>,
       cuda::std::chrono::duration<long, cuda::std::ratio<1, 1000>>,
       cuda::std::chrono::duration<long, cuda::std::ratio<1, 1000>>>();
  test<cuda::std::chrono::duration<long, cuda::std::ratio<1, 100>>,
       cuda::std::chrono::duration<int, cuda::std::ratio<1, 1000>>,
       cuda::std::chrono::duration<long, cuda::std::ratio<1, 1000>>>();
  test<cuda::std::chrono::duration<char, cuda::std::ratio<1, 30>>,
       cuda::std::chrono::duration<short, cuda::std::ratio<1, 1000>>,
       cuda::std::chrono::duration<int, cuda::std::ratio<1, 3000>>>();
  test<cuda::std::chrono::duration<double, cuda::std::ratio<21, 1>>,
       cuda::std::chrono::duration<short, cuda::std::ratio<15, 1>>,
       cuda::std::chrono::duration<double, cuda::std::ratio<3, 1>>>();

  return 0;
}
