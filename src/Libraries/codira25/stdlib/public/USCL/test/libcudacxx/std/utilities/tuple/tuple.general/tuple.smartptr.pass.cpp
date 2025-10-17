/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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

//  Tuples of smart pointers; based on bug #18350
//  auto_ptr doesn't have a copy constructor that takes a const &, but tuple does.

#include <uscl/std/__memory_>
#include <uscl/std/tuple>

#include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::tuple<cuda::std::unique_ptr<char>> up;
    // cuda::std::tuple<cuda::std::shared_ptr<char>> sp;
    // cuda::std::tuple<cuda::std::weak_ptr<char>> wp;
  }
  {
    cuda::std::tuple<cuda::std::unique_ptr<char[]>> up;
    // cuda::std::tuple<cuda::std::shared_ptr<char[]>> sp;
    // cuda::std::tuple<cuda::std::weak_ptr<char[]>> wp;
  }
  // Smart pointers of type 'T[N]' are not tested here since they are not
  // supported by the standard nor by libc++'s implementation.
  // See https://reviews.llvm.org/D21320 for more information.

  return 0;
}
