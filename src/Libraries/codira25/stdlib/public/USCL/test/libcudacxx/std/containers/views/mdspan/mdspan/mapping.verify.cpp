/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 8, 2022.
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

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right, class AccessorPolicy =
// default_accessor> class mdspan;
//
// Mandates:
//  - LayoutPolicy shall meet the layout mapping policy requirements ([mdspan.layout.policy.reqmts])

#include <uscl/std/mdspan>

#include "test_macros.h"

__host__ __device__ void not_layout_policy()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: LayoutPolicy template parameter
  // is invalid. A common mistake is to pass a layout mapping instead of a layout policy}}
  cuda::std::mdspan<int, cuda::std::extents<int>, cuda::std::layout_left::template mapping<cuda::std::extents<int>>> m;
  unused(m);
}

int main(int, char**)
{
  return 0;
}
