/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test forward

#include <uscl/std/utility>

#include "test_macros.h"

struct A
{};

__host__ __device__ A source()
{
  return A();
}
__host__ __device__ const A csource()
{
  return A();
}

int main(int, char**)
{
  {
    (void) cuda::std::forward<A&>(source()); // expected-note {{requested here}}
    // expected-error-re@__utility/forward.h:* {{{{(static_assert|static assertion)}} failed{{.*}} {{"?}}cannot forward
    // an rvalue as an lvalue{{"?}}}}
#if TEST_COMPILER(CLANG, >, 14)
    // expected-error {{ignoring return value of function declared with const attribute}}
#endif // TEST_COMPILER(CLANG, >, 14)
  }
  {
    const A ca = A();
    cuda::std::forward<A&>(ca); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    cuda::std::forward<A&>(csource()); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    const A ca = A();
    cuda::std::forward<A>(ca); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    cuda::std::forward<A>(csource()); // expected-error {{no matching function for call to 'forward'}}
  }
  {
    A a;
    cuda::std::forward(a); // expected-error {{no matching function for call to 'forward'}}
  }

  return 0;
}
