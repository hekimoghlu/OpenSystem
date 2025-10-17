/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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

// <new>

// template <class T> constexpr T* launder(T* p) noexcept;
// The program is ill-formed if T is a function type or cv void.

#include <uscl/std/__new_>
#include <uscl/std/cassert>

#include "test_macros.h"

__host__ __device__ void foo() {}

int main(int, char**)
{
  void* p = nullptr;
  (void) cuda::std::launder((void*) nullptr);
  (void) cuda::std::launder((const void*) nullptr);
  (void) cuda::std::launder((volatile void*) nullptr);
  (void) cuda::std::launder((const volatile void*) nullptr); // expected-error-re@new:* 4 {{static assertion
                                                             // failed{{.*}}can't launder cv-void}}
  // expected-error@new:* 0-4 {{void pointer argument to '__builtin_launder' is not allowed}}

  (void) cuda::std::launder(foo); // expected-error-re@new:* 1 {{static assertion failed{{.*}}can't launder functions}}
  // expected-error@new:* 0-1 {{function pointer argument to '__builtin_launder' is not allowed}}

  return 0;
}
