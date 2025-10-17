/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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

#ifndef ATOMIC_HELPERS_H
#define ATOMIC_HELPERS_H

#include <uscl/atomic>
#include <uscl/std/cassert>

#include "test_macros.h"

struct UserAtomicType
{
  int i;

  __host__ __device__ explicit UserAtomicType(int d = 0) noexcept
      : i(d)
  {}

  __host__ __device__ friend bool operator==(const UserAtomicType& x, const UserAtomicType& y)
  {
    return x.i == y.i;
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          >
struct TestEachIntegralType
{
  __host__ __device__ void operator()() const
  {
    TestFunctor<char, Selector, Scope>()();
    TestFunctor<signed char, Selector, Scope>()();
    TestFunctor<unsigned char, Selector, Scope>()();
    TestFunctor<short, Selector, Scope>()();
    TestFunctor<unsigned short, Selector, Scope>()();
    TestFunctor<int, Selector, Scope>()();
    TestFunctor<unsigned int, Selector, Scope>()();
    TestFunctor<long, Selector, Scope>()();
    TestFunctor<unsigned long, Selector, Scope>()();
    TestFunctor<long long, Selector, Scope>()();
    TestFunctor<unsigned long long, Selector, Scope>()();
    TestFunctor<wchar_t, Selector, Scope>();
    TestFunctor<char16_t, Selector, Scope>()();
    TestFunctor<char32_t, Selector, Scope>()();
    TestFunctor<int8_t, Selector, Scope>()();
    TestFunctor<uint8_t, Selector, Scope>()();
    TestFunctor<int16_t, Selector, Scope>()();
    TestFunctor<uint16_t, Selector, Scope>()();
    TestFunctor<int32_t, Selector, Scope>()();
    TestFunctor<uint32_t, Selector, Scope>()();
    TestFunctor<int64_t, Selector, Scope>()();
    TestFunctor<uint64_t, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          >
struct TestEachFloatingPointType
{
  __host__ __device__ void operator()() const
  {
    TestFunctor<float, Selector, Scope>()();
    TestFunctor<double, Selector, Scope>()();
  }
};

template <template <class, template <typename, typename> class, cuda::thread_scope> class TestFunctor,
          template <typename, typename> class Selector,
          cuda::thread_scope Scope
#if _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          = cuda::thread_scope_system
#endif // _CCCL_HOST_COMPILATION() || _CCCL_PTX_ARCH() >= 600
          >
struct TestEachAtomicType
{
  __host__ __device__ void operator()() const
  {
    TestEachIntegralType<TestFunctor, Selector, Scope>()();
    TestEachFloatingPointType<TestFunctor, Selector, Scope>()();
    TestFunctor<UserAtomicType, Selector, Scope>()();
    TestFunctor<int*, Selector, Scope>()();
    TestFunctor<const int*, Selector, Scope>()();
  }
};

#endif // ATOMIC_HELPER_H
