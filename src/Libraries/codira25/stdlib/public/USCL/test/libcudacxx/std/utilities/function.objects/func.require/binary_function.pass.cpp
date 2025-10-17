/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 19, 2022.
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

// <cuda/std/functional>
// REQUIRES: c++98 || c++03 || c++11 || c++14
// binary_function was removed in C++17

// binary_function

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <uscl/std/functional>
#include <uscl/std/type_traits>

int main(int, char**)
{
  typedef cuda::std::binary_function<int, short, bool> bf;
  static_assert((cuda::std::is_same<bf::first_argument_type, int>::value), "");
  static_assert((cuda::std::is_same<bf::second_argument_type, short>::value), "");
  static_assert((cuda::std::is_same<bf::result_type, bool>::value), "");

  return 0;
}
