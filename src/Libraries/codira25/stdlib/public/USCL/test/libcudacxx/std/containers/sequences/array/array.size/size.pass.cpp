/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <class T, size_t N> constexpr size_type array<T,N>::size();

#include <uscl/std/array>
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c = {1, 2, 3.5};
    assert(c.size() == 3);
    assert(c.max_size() == 3);
    assert(!c.empty());
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    C c = {};
    assert(c.size() == 0);
    assert(c.max_size() == 0);
    assert(c.empty());
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    constexpr C c = {1, 2, 3.5};
    static_assert(c.size() == 3, "");
    static_assert(c.max_size() == 3, "");
    static_assert(!c.empty(), "");
  }
  {
    typedef double T;
    typedef cuda::std::array<T, 0> C;
    constexpr C c = {};
    static_assert(c.size() == 0, "");
    static_assert(c.max_size() == 0, "");
    static_assert(c.empty(), "");
  }

  return 0;
}
