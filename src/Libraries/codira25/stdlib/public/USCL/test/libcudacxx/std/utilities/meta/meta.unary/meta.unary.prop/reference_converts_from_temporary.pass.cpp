/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

// is_const

#include <uscl/std/type_traits>

#include "test_macros.h"

#ifdef _CCCL_BUILTIN_REFERENCE_CONVERTS_FROM_TEMPORARY

struct SimpleClass
{
  SimpleClass() = default;
};
struct ConvertsToRvalue
{
  constexpr operator int();
  explicit constexpr operator int&&();
};
struct ConvertsToConstReference
{
  constexpr operator int();
  explicit constexpr operator int&();
};

// not references
static_assert(cuda::std::reference_converts_from_temporary<int, int>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<void, void>::value == false, "");

// references do not bind
static_assert(cuda::std::reference_converts_from_temporary<int&, int>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, int&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, int&&>::value == false, "");

// references do not bind to non-convertible types
static_assert(cuda::std::reference_converts_from_temporary<int&, void>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, const volatile void>::value == false, "");

// references do not bind to convertible types
static_assert(cuda::std::reference_converts_from_temporary<int&, long>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, long&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&, long&&>::value == false, "");

// const references bind to values
static_assert(cuda::std::reference_converts_from_temporary<const int&, int>::value == true, "");

// const references do not bind to other references
static_assert(cuda::std::reference_converts_from_temporary<const int&, int&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, int&&>::value == false, "");

// const references bind to converted values
static_assert(cuda::std::reference_converts_from_temporary<const int&, long>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, long&>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, long&&>::value == true, "");

// rvalue references behave similar to const lvalue references
static_assert(cuda::std::reference_converts_from_temporary<int&&, int>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, int&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, int&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, long>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, long&>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, long&&>::value == true, "");

// simple class types behave like builtin types
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&, SimpleClass>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&, SimpleClass&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&, SimpleClass&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const SimpleClass&, SimpleClass>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&&, SimpleClass>::value == true, "");

// No conversion possible
static_assert(cuda::std::reference_converts_from_temporary<const SimpleClass&, SimpleClass&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const SimpleClass&, SimpleClass&&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&&, SimpleClass&>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<SimpleClass&&, SimpleClass&&>::value == false, "");

// arrays do not bind to references
static_assert(cuda::std::reference_converts_from_temporary<int&, int[]>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, int[]>::value == false, "");
static_assert(cuda::std::reference_converts_from_temporary<int&&, int[]>::value == false, "");

// In contrast to reference_constructs_from_temporary conversions are possible
static_assert(cuda::std::reference_converts_from_temporary<int&&, ConvertsToRvalue>::value == true, "");
static_assert(cuda::std::reference_converts_from_temporary<const int&, ConvertsToConstReference>::value == true, "");

#endif // _CCCL_BUILTIN_REFERENCE_CONVERTS_FROM_TEMPORARY

int main(int, char**)
{
  return 0;
}
