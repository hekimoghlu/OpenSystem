/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// LWG issue 3024

#include <uscl/std/type_traits>
#include <uscl/std/variant>

struct NotCopyConstructible
{
  NotCopyConstructible()                            = default;
  NotCopyConstructible(NotCopyConstructible const&) = delete;
};

int main(int, char**)
{
  static_assert(!cuda::std::is_copy_constructible_v<NotCopyConstructible>);

  cuda::std::variant<NotCopyConstructible> v;
  cuda::std::variant<NotCopyConstructible> v1;
  cuda::std::variant<NotCopyConstructible> v2(v); // expected-error {{call to implicitly-deleted copy constructor of
                                                  // 'cuda::std::variant<NotCopyConstructible>'}}
  v1 = v; // expected-error-re {{object of type 'std:{{.*}}:variant<NotCopyConstructible>' cannot be assigned because
          // its copy assignment operator is implicitly deleted}}
}
