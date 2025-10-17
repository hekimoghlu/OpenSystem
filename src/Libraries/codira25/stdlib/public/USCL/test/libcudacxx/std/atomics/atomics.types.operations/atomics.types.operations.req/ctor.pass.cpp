/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 9, 2023.
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
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// constexpr atomic<T>::atomic(T value)

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <uscl/atomic>
#include <uscl/std/atomic>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

struct UserType
{
  int i;

  __host__ __device__ UserType() noexcept {}
  __host__ __device__ constexpr explicit UserType(int d) noexcept
      : i(d)
  {}

  __host__ __device__ friend bool operator==(const UserType& x, const UserType& y)
  {
    return x.i == y.i;
  }
};

template <class Tp, template <typename, typename> class, cuda::thread_scope Scope>
struct TestFunc
{
  __host__ __device__ void operator()() const
  {
    typedef cuda::atomic<Tp, Scope> Atomic;
    static_assert(cuda::std::is_literal_type<Atomic>::value, "");
    constexpr Tp t(42);
    {
      constexpr Atomic a(t);
      assert(a == t);
    }
    {
      constexpr Atomic a{t};
      assert(a == t);
    }
#if !_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >=, 4, 9)
    // TODO: Figure out why this is failing with GCC 4.8.2 on CentOS 7 only.
    {
      constexpr Atomic a = LIBCUDACXX_ATOMIC_VAR_INIT(t);
      assert(a == t);
    }
#endif
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (TestFunc<UserType, local_memory_selector, cuda::thread_scope_system>()();
     TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_system>()();),
    NV_PROVIDES_SM_60,
    (TestFunc<UserType, local_memory_selector, cuda::thread_scope_system>()();
     TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_system>()();))

  TestFunc<UserType, local_memory_selector, cuda::thread_scope_device>()();
  TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_device>()();
  TestFunc<UserType, local_memory_selector, cuda::thread_scope_block>()();
  TestEachIntegralType<TestFunc, local_memory_selector, cuda::thread_scope_block>()();

  return 0;
}
