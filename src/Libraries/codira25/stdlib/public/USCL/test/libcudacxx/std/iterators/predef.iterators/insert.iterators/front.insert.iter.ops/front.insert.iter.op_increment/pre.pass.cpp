/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 19, 2021.
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

// <cuda/std/iterator>

// front_insert_iterator

// front_insert_iterator<Cont>& operator++();

#include <uscl/std/cassert>
#include <uscl/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>

#  include "nasty_containers.h"
#endif // _LIBCUDACXX_HAS_LIST

#include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
{
  cuda::std::front_insert_iterator<C> i(c);
  cuda::std::front_insert_iterator<C>& r = ++i;
  assert(&r == &i);
}

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_LIST)
  test(cuda::std::list<int>());
  test(nasty_list<int>());
#endif

  return 0;
}
