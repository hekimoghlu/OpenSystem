/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

// <cuda/atomic>

#include <uscl/atomic>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T,
          template <typename, typename> class Selector,
          cuda::thread_scope ThreadScope,
          bool Signed = cuda::std::is_signed<T>::value>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    // Test greater
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(1);
      assert(t.fetch_max(2) == T(1));
      assert(t.load() == T(2));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(1);
      assert(t.fetch_max(2) == T(1));
      assert(t.load() == T(2));
    }
    // Test not greater
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(3);
      assert(t.fetch_max(2) == T(3));
      assert(t.load() == T(3));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(3);
      assert(t.fetch_max(2) == T(3));
      assert(t.load() == T(3));
    }
  }
};

template <class T, template <typename, typename> class Selector, cuda::thread_scope ThreadScope>
struct TestFn<T, Selector, ThreadScope, true>
{
  __host__ __device__ void operator()() const
  {
    // Call unsigned tests
    TestFn<T, Selector, ThreadScope, false>()();
    // Test greater, but with signed math
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-5);
      assert(t.fetch_max(-1) == T(-5));
      assert(t.load() == T(-1));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-5);
      assert(t.fetch_max(-1) == T(-5));
      assert(t.load() == T(-1));
    }
    // Test not greater
    {
      typedef cuda::atomic<T> A;
      Selector<A, constructor_initializer> sel;
      A& t = *sel.construct();
      t    = T(-1);
      assert(t.fetch_max(-5) == T(-1));
      assert(t.load() == T(-1));
    }
    {
      typedef cuda::atomic<T> A;
      Selector<volatile A, constructor_initializer> sel;
      volatile A& t = *sel.construct();
      t             = T(-1);
      assert(t.fetch_max(-5) == T(-1));
      assert(t.load() == T(-1));
    }
  }
};

template <class T, template <typename, typename> class Selector, cuda::thread_scope ThreadScope>
struct TestFnDispatch
{
  __host__ __device__ void operator()() const
  {
    TestFn<T, Selector, ThreadScope>()();
  }
};

int main(int, char**)
{
  NV_DISPATCH_TARGET(
    NV_IS_HOST,
    (TestEachIntegralType<TestFnDispatch, local_memory_selector>()();
     TestEachFloatingPointType<TestFnDispatch, local_memory_selector>()();),
    NV_PROVIDES_SM_70,
    (TestEachIntegralType<TestFnDispatch, local_memory_selector>()();
     TestEachFloatingPointType<TestFnDispatch, local_memory_selector>()();))

  NV_IF_TARGET(NV_IS_DEVICE,
               (TestEachIntegralType<TestFnDispatch, shared_memory_selector>()();
                TestEachFloatingPointType<TestFnDispatch, shared_memory_selector>()();
                TestEachIntegralType<TestFnDispatch, global_memory_selector>()();
                TestEachFloatingPointType<TestFnDispatch, global_memory_selector>()();))

  return 0;
}
