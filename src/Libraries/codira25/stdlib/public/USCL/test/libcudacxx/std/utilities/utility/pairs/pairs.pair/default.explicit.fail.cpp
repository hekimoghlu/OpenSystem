/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 4, 2022.
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

// Before GCC 6, aggregate initialization kicks in.
// See https://stackoverflow.com/q/41799015/627587.
// UNSUPPORTED: gcc-5

// <utility>

// template <class T1, class T2> struct pair

// explicit(see-below) constexpr pair();

// This test checks the conditional explicitness of cuda::std::pair's default
// constructor as introduced by the resolution of LWG 2510.

#include <uscl/std/utility>

struct ImplicitlyDefaultConstructible
{
  ImplicitlyDefaultConstructible() = default;
};

struct ExplicitlyDefaultConstructible
{
  explicit ExplicitlyDefaultConstructible() = default;
};

cuda::std::pair<ImplicitlyDefaultConstructible, ExplicitlyDefaultConstructible> test1()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::pair<ExplicitlyDefaultConstructible, ImplicitlyDefaultConstructible> test2()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::pair<ExplicitlyDefaultConstructible, ExplicitlyDefaultConstructible> test3()
{
  return {};
} // expected-error 1 {{chosen constructor is explicit in copy-initialization}}
cuda::std::pair<ImplicitlyDefaultConstructible, ImplicitlyDefaultConstructible> test4()
{
  return {};
}

int main(int, char**)
{
  return 0;
}
