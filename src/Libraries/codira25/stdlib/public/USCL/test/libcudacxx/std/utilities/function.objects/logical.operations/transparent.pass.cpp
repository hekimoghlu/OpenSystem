/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
  static_assert(!is_transparent<cuda::std::logical_and<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::logical_and<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::logical_and<void>>::value, "");
  static_assert(is_transparent<cuda::std::logical_and<>>::value, "");

  static_assert(!is_transparent<cuda::std::logical_or<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::logical_or<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::logical_or<void>>::value, "");
  static_assert(is_transparent<cuda::std::logical_or<>>::value, "");

  static_assert(!is_transparent<cuda::std::logical_not<int>>::value, "");
  // static_assert ( !is_transparent<cuda::std::logical_not<cuda::std::string>>::value, "" );
  static_assert(is_transparent<cuda::std::logical_not<void>>::value, "");
  static_assert(is_transparent<cuda::std::logical_not<>>::value, "");

  return 0;
}
