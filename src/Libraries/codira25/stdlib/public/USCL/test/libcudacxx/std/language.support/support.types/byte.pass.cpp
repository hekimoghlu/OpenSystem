/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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

#include <uscl/std/cstddef>
#include <uscl/std/type_traits>

#include "test_macros.h"

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

// cuda::std::byte is not an integer type, nor a character type.
// It is a distinct type for accessing the bits that ultimately make up object storage.

static_assert(cuda::std::is_trivial<cuda::std::byte>::value, "");
static_assert(!cuda::std::is_arithmetic<cuda::std::byte>::value, "");
static_assert(!cuda::std::is_integral<cuda::std::byte>::value, "");

static_assert(!cuda::std::is_same<cuda::std::byte, char>::value, "");
static_assert(!cuda::std::is_same<cuda::std::byte, signed char>::value, "");
static_assert(!cuda::std::is_same<cuda::std::byte, unsigned char>::value, "");

// The standard doesn't outright say this, but it's pretty clear that it has to be true.
static_assert(sizeof(cuda::std::byte) == 1, "");

int main(int, char**)
{
  return 0;
}
