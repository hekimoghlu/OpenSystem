/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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

// template<class NotAnIterator>
// struct iterator_traits
// {
// };

#include <uscl/std/iterator>

#include "test_macros.h"

struct not_an_iterator
{};

template <class T>
struct has_value_type
{
private:
  struct two
  {
    char lx;
    char lxx;
  };

  template <class U>
  __host__ __device__ static two test(...);

  template <class U>
  __host__ __device__ static char test(typename U::value_type* = 0);

public:
  static const bool value = sizeof(test<T>(0)) == 1;
};

int main(int, char**)
{
  typedef cuda::std::iterator_traits<not_an_iterator> It;
  static_assert(!(has_value_type<It>::value), "");

  return 0;
}
