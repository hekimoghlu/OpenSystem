/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
//

// <cuda/std/type_traits>

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

using cuda::std::__copy_cvref_t;
using cuda::std::is_same;

// Ensure that we copy the proper qualifiers
static_assert(is_same<float, __copy_cvref_t<int, float>>::value, "");
static_assert(is_same<const float, __copy_cvref_t<const int, float>>::value, "");
static_assert(is_same<volatile float, __copy_cvref_t<volatile int, float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, float>>::value, "");

static_assert(is_same<const float, __copy_cvref_t<int, const float>>::value, "");
static_assert(is_same<const float, __copy_cvref_t<const int, const float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<volatile int, const float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, const float>>::value, "");

static_assert(is_same<volatile float, __copy_cvref_t<int, volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const int, volatile float>>::value, "");
static_assert(is_same<volatile float, __copy_cvref_t<volatile int, volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, volatile float>>::value, "");

static_assert(is_same<const volatile float, __copy_cvref_t<int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<volatile int, const volatile float>>::value, "");
static_assert(is_same<const volatile float, __copy_cvref_t<const volatile int, const volatile float>>::value, "");

// Ensure that we do copy lvalue-reference qualifiers to types without reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&, float>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, float>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, float>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&, const float>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, const float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const float>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&, volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, volatile float>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, volatile float>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const volatile float>>::value, "");

// Ensure that we do copy lvalue-reference qualifiers to types with lvalue-reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<volatile int&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const volatile int&, float&>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<volatile int&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const volatile int&, const float&>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const volatile int&, volatile float&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const volatile float&>>::value, "");

// Ensure that we do copy lvalue-reference qualifiers to types rvalue-reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&, float&&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const int&, float&&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<volatile int&, float&&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const volatile int&, float&&>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&, const float&&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&, const float&&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<volatile int&, const float&&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const volatile int&, const float&&>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const volatile int&, volatile float&&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&, const volatile float&&>>::value, "");

// Ensure that we do copy rvalue-reference qualifiers to types without reference qualifiers
static_assert(is_same<float&&, __copy_cvref_t<int&&, float>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const int&&, float>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<volatile int&&, float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, float>>::value, "");

static_assert(is_same<const float&&, __copy_cvref_t<int&&, const float>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const int&&, const float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<volatile int&&, const float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, const float>>::value, "");

static_assert(is_same<volatile float&&, __copy_cvref_t<int&&, volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const int&&, volatile float>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<volatile int&&, volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, volatile float>>::value, "");

static_assert(is_same<const volatile float&&, __copy_cvref_t<int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<volatile int&&, const volatile float>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, const volatile float>>::value, "");

// Ensure that we do not copy rvalue-reference qualifiers to types with lvalue-reference qualifiers
static_assert(is_same<float&, __copy_cvref_t<int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<volatile int&&, float&>>::value, "");
static_assert(is_same<float&, __copy_cvref_t<const volatile int&&, float&>>::value, "");

static_assert(is_same<const float&, __copy_cvref_t<int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<volatile int&&, const float&>>::value, "");
static_assert(is_same<const float&, __copy_cvref_t<const volatile int&&, const float&>>::value, "");

static_assert(is_same<volatile float&, __copy_cvref_t<int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<volatile int&&, volatile float&>>::value, "");
static_assert(is_same<volatile float&, __copy_cvref_t<const volatile int&&, volatile float&>>::value, "");

static_assert(is_same<const volatile float&, __copy_cvref_t<int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<volatile int&&, const volatile float&>>::value, "");
static_assert(is_same<const volatile float&, __copy_cvref_t<const volatile int&&, const volatile float&>>::value, "");

// Ensure that we do keep rvalue-reference qualifiers to types with rvalue-reference qualifiers
static_assert(is_same<float&&, __copy_cvref_t<int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cvref_t<const int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cvref_t<volatile int&&, float&&>>::value, "");
static_assert(is_same<float&&, __copy_cvref_t<const volatile int&&, float&&>>::value, "");

static_assert(is_same<const float&&, __copy_cvref_t<int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<volatile int&&, const float&&>>::value, "");
static_assert(is_same<const float&&, __copy_cvref_t<const volatile int&&, const float&&>>::value, "");

static_assert(is_same<volatile float&&, __copy_cvref_t<int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<const int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<volatile int&&, volatile float&&>>::value, "");
static_assert(is_same<volatile float&&, __copy_cvref_t<const volatile int&&, volatile float&&>>::value, "");

static_assert(is_same<const volatile float&&, __copy_cvref_t<int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<volatile int&&, const volatile float&&>>::value, "");
static_assert(is_same<const volatile float&&, __copy_cvref_t<const volatile int&&, const volatile float&&>>::value, "");

int main(int, char**)
{
  return 0;
}
