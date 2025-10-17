/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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

// __cccl_is_referenceable<Tp>
//
// [defns.referenceable] defines "a referenceable type" as:
// An object type, a function type that does not have cv-qualifiers
//    or a ref-qualifier, or a reference type.
//

#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

struct Foo
{};

static_assert((!cuda::std::__cccl_is_referenceable<void>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<int>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<int[3]>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<int[]>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<int&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<const int&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<int*>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<const int*>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<Foo>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<const Foo>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<Foo&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<const Foo&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<Foo&&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<const Foo&&>::value), "");

#if !TEST_COMPILER(MSVC)
static_assert((cuda::std::__cccl_is_referenceable<int __attribute__((__vector_size__(8)))>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<const int __attribute__((__vector_size__(8)))>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<float __attribute__((__vector_size__(16)))>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<const float __attribute__((__vector_size__(16)))>::value), "");
#endif // !TEST_COMPILER(MSVC)

// Functions without cv-qualifiers are referenceable
static_assert((cuda::std::__cccl_is_referenceable<void()>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void() const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void() &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void() const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void() &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void() const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void(int)>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int) const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int) &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int) const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int) &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void(int, float)>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float) const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float) &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float) const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float) &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void(int, float, Foo&)>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&) const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&) &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&) const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&) &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void(...)>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(...) const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(...) &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(...) const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(...) &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(...) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void(int, ...)>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, ...) const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, ...) &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, ...) const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, ...) &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, ...) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void(int, float, ...)>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, ...) const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, ...) &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, ...) const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, ...) &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, ...) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void(int, float, Foo&, ...)>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&, ...) const>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&, ...) &>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&, ...) const&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&, ...) &&>::value), "");
static_assert((!cuda::std::__cccl_is_referenceable<void(int, float, Foo&, ...) const&&>::value), "");

// member functions with or without cv-qualifiers are referenceable
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)()>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)() const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)() &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)() const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)() &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)() const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int)>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int) const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int) &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int) const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int) &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float)>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float) const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float) &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float) const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float) &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&)>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&) const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&) &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&) const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&) &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(...)>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(...) const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(...) &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(...) const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(...) &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(...) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, ...)>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, ...) const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, ...) &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, ...) const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, ...) &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, ...) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, ...)>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, ...) const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, ...) &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, ...) const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, ...) &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, ...) const&&>::value), "");

static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&, ...)>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&, ...) const>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&, ...) &>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&, ...) const&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&, ...) &&>::value), "");
static_assert((cuda::std::__cccl_is_referenceable<void (Foo::*)(int, float, Foo&, ...) const&&>::value), "");

int main(int, char**)
{
  return 0;
}
