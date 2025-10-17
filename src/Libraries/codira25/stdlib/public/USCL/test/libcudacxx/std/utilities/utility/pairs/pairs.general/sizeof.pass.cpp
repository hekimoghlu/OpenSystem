/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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

#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(sizeof(cuda::std::pair<float, int>) == sizeof(float) + sizeof(int));
  static_assert(sizeof(cuda::std::pair<cuda::std::pair<float, int>, cuda::std::pair<float, int>>)
                == 2 * sizeof(cuda::std::pair<float, int>));
  static_assert(sizeof(cuda::std::pair<cuda::std::pair<float, int>, cuda::std::pair<float, int>>)
                == sizeof(float) * 2 + sizeof(int) * 2);

  return 0;
}
