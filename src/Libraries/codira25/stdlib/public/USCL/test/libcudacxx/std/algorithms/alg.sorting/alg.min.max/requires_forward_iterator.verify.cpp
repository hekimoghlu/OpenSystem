/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter>
//   max_element(Iter first, Iter last);

#include <uscl/std/__algorithm_>
#include <uscl/std/cassert>

#include "test_iterators.h"

int main(int, char**)
{
  int arr[]    = {1, 2, 3};
  const int *b = cuda::std::begin(arr), *e = cuda::std::end(arr);
  typedef cpp17_input_iterator<const int*> Iter;
  {
    // expected-error@*:* {{cuda::std::min_element requires a ForwardIterator}}
    (void) cuda::std::min_element(Iter(b), Iter(e));
  }
  {
    // expected-error@*:* {{cuda::std::max_element requires a ForwardIterator}}
    (void) cuda::std::max_element(Iter(b), Iter(e));
  }
  {
    // expected-error@*:* {{cuda::std::minmax_element requires a ForwardIterator}}
    (void) cuda::std::minmax_element(Iter(b), Iter(e));
  }

  return 0;
}
