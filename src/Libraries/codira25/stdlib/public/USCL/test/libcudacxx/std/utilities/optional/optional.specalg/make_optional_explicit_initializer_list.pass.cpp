/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 1, 2023.
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

// <cuda/std/optional>

// template <class T, class U, class... Args>
//   constexpr optional<T> make_optional(initializer_list<U> il, Args&&... args);

#include <uscl/std/cassert>
// #include <uscl/std/memory>
#include <uscl/std/optional>
// #include <uscl/std/string>

#include "test_macros.h"

struct TestT
{
  int x;
  int size;
  int* ptr;
  __host__ __device__ constexpr TestT(cuda::std::initializer_list<int> il)
      : x(*il.begin())
      , size(static_cast<int>(il.size()))
      , ptr(nullptr)
  {}
  __host__ __device__ constexpr TestT(cuda::std::initializer_list<int> il, int* p)
      : x(*il.begin())
      , size(static_cast<int>(il.size()))
      , ptr(p)
  {}
};

__host__ __device__ constexpr bool test()
{
  {
    auto opt = cuda::std::make_optional<TestT>({42, 2, 3});
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<TestT>>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == nullptr);
  }
  {
    int i    = 42;
    auto opt = cuda::std::make_optional<TestT>({42, 2, 3}, &i);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<TestT>>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == &i);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_ADDRESSOF
  /*
  {
    auto opt = cuda::std::make_optional<cuda::std::string>({'1', '2', '3'});
    assert(*opt == "123");
  }
  {
    auto opt = cuda::std::make_optional<cuda::std::string>({'a', 'b', 'c'}, cuda::std::allocator<char>{});
    assert(*opt == "abc");
  }
  */
  return 0;
}
