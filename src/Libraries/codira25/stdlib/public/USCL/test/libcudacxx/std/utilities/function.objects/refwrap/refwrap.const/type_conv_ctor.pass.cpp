/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 9, 2022.
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

// <functional>
//
// reference_wrapper
//
// template <class U>
//   reference_wrapper(U&&) noexcept(see below);

// #include <uscl/std/functional>
#include <uscl/std/cassert>
#include <uscl/std/utility>

#include "test_macros.h"

struct convertible_to_int_ref
{
  int val = 0;
  __host__ __device__ operator int&()
  {
    return val;
  }
  __host__ __device__ operator int const&() const
  {
    return val;
  }
};

template <bool IsNothrow>
struct nothrow_convertible
{
  int val = 0;
  __host__ __device__ operator int&() noexcept(IsNothrow)
  {
    return val;
  }
};

struct convertible_from_int
{
  __host__ __device__ convertible_from_int(int) {}
};

__host__ __device__ void meow(cuda::std::reference_wrapper<int>) {}
__host__ __device__ void meow(convertible_from_int) {}

int main(int, char**)
{
  {
    convertible_to_int_ref t;
    cuda::std::reference_wrapper<convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    const convertible_to_int_ref t{};
    cuda::std::reference_wrapper<const convertible_to_int_ref> r(t);
    assert(&r.get() == &t);
  }
  {
    using Ref = cuda::std::reference_wrapper<int>;
    static_assert(noexcept(Ref(nothrow_convertible<true>())));
#if !TEST_COMPILER(NVHPC)
    static_assert(!noexcept(Ref(nothrow_convertible<false>())));
#endif // !TEST_COMPILER(NVHPC)
  }
  {
    meow(0);
  }
#if !TEST_COMPILER(MSVC) && !TEST_COMPILER(NVRTC)
  {
    extern cuda::std::reference_wrapper<int> purr();
    static_assert(cuda::std::is_same_v<decltype(true ? purr() : 0), int>);
  }
#endif // !TEST_COMPILER(MSVC)
#if !TEST_COMPILER(GCC, <, 8) // gcc-7 is broken wrt ctad
  {
    int i = 0;
    cuda::std::reference_wrapper ri(i);
    static_assert((cuda::std::is_same<decltype(ri), cuda::std::reference_wrapper<int>>::value), "");
    const int j = 0;
    cuda::std::reference_wrapper rj(j);
    static_assert((cuda::std::is_same<decltype(rj), cuda::std::reference_wrapper<const int>>::value), "");
  }
#endif // !TEST_COMPILER(GCC, <, 8)

  return 0;
}
