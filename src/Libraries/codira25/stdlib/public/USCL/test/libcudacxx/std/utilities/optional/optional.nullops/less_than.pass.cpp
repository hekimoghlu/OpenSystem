/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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

// template <class T> constexpr bool operator<(const optional<T>& x, nullopt_t) noexcept;
// template <class T> constexpr bool operator<(nullopt_t, const optional<T>& x) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/optional>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test()
{
  using cuda::std::nullopt;
  using cuda::std::nullopt_t;
  using cuda::std::optional;

  {
    using O = optional<T>;
    cuda::std::remove_reference_t<T> val{1};

    O o1{}; // disengaged
    O o2{val}; // engaged

    assert(!(nullopt < o1));
    assert((nullopt < o2));
    assert(!(o1 < nullopt));
    assert(!(o2 < nullopt));

    static_assert(noexcept(nullopt < o1), "");
    static_assert(noexcept(o1 < nullopt), "");
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
