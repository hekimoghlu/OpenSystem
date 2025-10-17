/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 19, 2023.
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

// template <class T>
//     bool
//     atomic_is_lock_free(const volatile atomic<T>* obj);
//
// template <class T>
//     bool
//     atomic_is_lock_free(const atomic<T>* obj);

#include <uscl/std/atomic>
#include <uscl/std/cassert>

#include "atomic_helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T, template <typename, typename> class, cuda::thread_scope>
struct TestFn
{
  __host__ __device__ void operator()() const
  {
    typedef cuda::std::atomic<T> A;
    A t{};
    bool b1 = cuda::std::atomic_is_lock_free(static_cast<const A*>(&t));
    volatile A vt{};
    bool b2 = cuda::std::atomic_is_lock_free(static_cast<const volatile A*>(&vt));
    assert(b1 == b2);
  }
};

struct A
{
  char _[4];
};

int main(int, char**)
{
  TestFn<A, local_memory_selector, cuda::thread_scope_system>()();
  TestEachAtomicType<TestFn, local_memory_selector>()();

  return 0;
}
