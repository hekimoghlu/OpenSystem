/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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

// template <class T> constexpr size_t tuple_size_v = tuple_size<T>::value;

// Expect failures with a reference type, pointer type, and a non-tuple type.

#include <uscl/std/tuple>

int main(int, char**)
{
  (void) cuda::std::tuple_size_v<cuda::std::tuple<>&>; // expected-note {{requested here}}
  (void) cuda::std::tuple_size_v<int>; // expected-note {{requested here}}
  (void) cuda::std::tuple_size_v<cuda::std::tuple<>*>; // expected-note {{requested here}}
  // expected-error@tuple:* 3 {{implicit instantiation of undefined template}}

  return 0;
}
