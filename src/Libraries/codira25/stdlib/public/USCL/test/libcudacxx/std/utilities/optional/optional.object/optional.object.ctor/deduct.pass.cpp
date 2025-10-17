/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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

//
// GCC 7 seems to be extremely broken when it comes to deduction here.
// It fails the test on all the ctor calls, but if the template argument to optional
// is provided explicitly instead of being deduced, it compiles the test fine.
// Given that the deduction guide is trivial, this appears to be a compiler bug,
// so just don't run this test on GCC 7.
// UNSUPPORTED: gcc-6, gcc-7

// template<class T>
//   optional(T) -> optional<T>;

#include <uscl/std/cassert>
#include <uscl/std/optional>

#include "test_macros.h"

struct A
{};

int main(int, char**)
{
  //  Test the explicit deduction guides
  {
    //  optional(T)
    cuda::std::optional opt(5);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<int>>);
    assert(static_cast<bool>(opt));
    assert(*opt == 5);
  }

  {
    //  optional(T)
    cuda::std::optional opt(A{});
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<A>>);
    assert(static_cast<bool>(opt));
  }

  {
    //  optional(const T&);
    const int& source = 5;
    cuda::std::optional opt(source);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<int>>);
    assert(static_cast<bool>(opt));
    assert(*opt == 5);
  }

  {
    //  optional(T*);
    const int* source = nullptr;
    cuda::std::optional opt(source);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<const int*>>);
    assert(static_cast<bool>(opt));
    assert(*opt == nullptr);
  }

  {
    //  optional(T[]);
    int source[] = {1, 2, 3};
    cuda::std::optional opt(source);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<int*>>);
    assert(static_cast<bool>(opt));
    assert((*opt)[0] == 1);
  }

  //  Test the implicit deduction guides
  {
    //  optional(optional);
    cuda::std::optional<char> source('A');
    cuda::std::optional opt(source);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<char>>);
    assert(static_cast<bool>(opt) == static_cast<bool>(source));
    assert(*opt == *source);
  }

  return 0;
}
