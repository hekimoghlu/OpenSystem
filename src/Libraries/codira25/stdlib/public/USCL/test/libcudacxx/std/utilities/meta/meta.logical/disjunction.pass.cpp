/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 1, 2022.
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

// template<class... B> struct disjunction;                           // C++17
// template<class... B>
//   constexpr bool disjunction_v = disjunction<B...>::value;         // C++17

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

struct True
{
  static constexpr bool value = true;
};
struct False
{
  static constexpr bool value = false;
};

int main(int, char**)
{
  static_assert(!cuda::std::disjunction<>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type>::value, "");
  static_assert(!cuda::std::disjunction<cuda::std::false_type>::value, "");

  static_assert(!cuda::std::disjunction_v<>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type>, "");
  static_assert(!cuda::std::disjunction_v<cuda::std::false_type>, "");

  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::disjunction<cuda::std::false_type, cuda::std::false_type>::value, "");

  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::disjunction_v<cuda::std::false_type, cuda::std::false_type>, "");

  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::true_type, cuda::std::false_type, cuda::std::false_type>::value, "");
  static_assert(cuda::std::disjunction<cuda::std::false_type, cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(!cuda::std::disjunction<cuda::std::false_type, cuda::std::false_type, cuda::std::false_type>::value,
                "");

  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::true_type, cuda::std::false_type, cuda::std::false_type>, "");
  static_assert(cuda::std::disjunction_v<cuda::std::false_type, cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(!cuda::std::disjunction_v<cuda::std::false_type, cuda::std::false_type, cuda::std::false_type>, "");

  static_assert(cuda::std::disjunction<True>::value, "");
  static_assert(!cuda::std::disjunction<False>::value, "");

  static_assert(cuda::std::disjunction_v<True>, "");
  static_assert(!cuda::std::disjunction_v<False>, "");

  return 0;
}
