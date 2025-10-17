/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Tuple, __tuple_convertible<Tuple, tuple> >
//   tuple(Tuple &&);
//
// template <class Tuple, __tuple_constructible<Tuple, tuple> >
//   tuple(Tuple &&);

// This test checks that we do not evaluate __make_tuple_types
// on the array.

// cuda::std::array not supported
// #include <uscl/std/array>
#include <uscl/std/tuple>

#include "test_macros.h"

// cuda::std::array not supported
/*
// Use 1256 to try and blow the template instantiation depth for all compilers.
using array_t = cuda::std::array<char, 1256>;
using tuple_t = cuda::std::tuple<array_t>;
*/

int main(int, char**)
{
  /*
  array_t arr;
  tuple_t tup(arr);
  */
  return 0;
}
