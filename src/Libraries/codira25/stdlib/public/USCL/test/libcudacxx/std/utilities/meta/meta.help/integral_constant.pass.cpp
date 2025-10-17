/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 1, 2024.
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

// type_traits

// integral_constant

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::integral_constant<int, 5> _5;
  static_assert(_5::value == 5, "");
  static_assert((cuda::std::is_same<_5::value_type, int>::value), "");
  static_assert((cuda::std::is_same<_5::type, _5>::value), "");
  static_assert((_5() == 5), "");
  assert(_5() == 5);

  static_assert(_5{}() == 5, "");
  static_assert(cuda::std::true_type{}(), "");

  static_assert(cuda::std::false_type::value == false, "");
  static_assert((cuda::std::is_same<cuda::std::false_type::value_type, bool>::value), "");
  static_assert((cuda::std::is_same<cuda::std::false_type::type, cuda::std::false_type>::value), "");

  static_assert(cuda::std::true_type::value == true, "");
  static_assert((cuda::std::is_same<cuda::std::true_type::value_type, bool>::value), "");
  static_assert((cuda::std::is_same<cuda::std::true_type::type, cuda::std::true_type>::value), "");

  cuda::std::false_type f1;
  cuda::std::false_type f2 = f1;
  assert(!f2);

  cuda::std::true_type t1;
  cuda::std::true_type t2 = t1;
  assert(t2);

  return 0;
}
