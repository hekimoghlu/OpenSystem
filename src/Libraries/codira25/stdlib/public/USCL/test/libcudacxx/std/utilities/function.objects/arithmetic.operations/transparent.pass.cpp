/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 25, 2023.
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

#include <uscl/std/functional>
// #include <uscl/std/string>

template <class T>
struct is_transparent
{
private:
  struct two
  {
    char lx;
    char lxx;
  };
  template <class U>
  __host__ __device__ static two test(...);
  template <class U>
  __host__ __device__ static char test(typename U::is_transparent* = 0);

public:
  static const bool value = sizeof(test<T>(0)) == 1;
};

int main(int, char**)
{
  static_assert(!is_transparent<cuda::std::plus<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::plus<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::plus<void>>::value, "");
  static_assert(is_transparent<cuda::std::plus<>>::value, "");

  static_assert(!is_transparent<cuda::std::minus<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::minus<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::minus<void>>::value, "");
  static_assert(is_transparent<cuda::std::minus<>>::value, "");

  static_assert(!is_transparent<cuda::std::multiplies<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::multiplies<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::multiplies<void>>::value, "");
  static_assert(is_transparent<cuda::std::multiplies<>>::value, "");

  static_assert(!is_transparent<cuda::std::divides<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::divides<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::divides<void>>::value, "");
  static_assert(is_transparent<cuda::std::divides<>>::value, "");

  static_assert(!is_transparent<cuda::std::modulus<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::modulus<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::modulus<void>>::value, "");
  static_assert(is_transparent<cuda::std::modulus<>>::value, "");

  static_assert(!is_transparent<cuda::std::negate<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::negate<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::negate<void>>::value, "");
  static_assert(is_transparent<cuda::std::negate<>>::value, "");

  return 0;
}
