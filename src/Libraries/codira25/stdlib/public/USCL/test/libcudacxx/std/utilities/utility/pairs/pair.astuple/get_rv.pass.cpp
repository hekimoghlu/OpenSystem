/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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

// template<size_t I, class T1, class T2>
//     typename tuple_element<I, cuda::std::pair<T1, T2> >::type&&
//     get(pair<T1, T2>&&);

#include <uscl/std/utility>
// cuda/std/memory not supported
// #include <uscl/std/memory>
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  // cuda/std/memory not supported
  /*
  {
      typedef cuda::std::pair<cuda::std::unique_ptr<int>, short> P;
      P p(cuda::std::unique_ptr<int>(new int(3)), static_cast<short>(4));
      cuda::std::unique_ptr<int> ptr = cuda::std::get<0>(cuda::std::move(p));
      assert(*ptr == 3);
  }
  */
  return 0;
}
