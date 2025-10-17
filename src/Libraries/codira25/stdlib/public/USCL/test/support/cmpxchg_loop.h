/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#include <uscl/std/atomic>

template <class A, class T>
__host__ __device__ bool cmpxchg_weak_loop(A& atomic, T& expected, T desired)
{
  for (int i = 0; i < 10; i++)
  {
    if (atomic.compare_exchange_weak(expected, desired) == true)
    {
      return true;
    }
  }

  return false;
}

template <class A, class T>
__host__ __device__ bool
cmpxchg_weak_loop(A& atomic, T& expected, T desired, cuda::std::memory_order success, cuda::std::memory_order failure)
{
  for (int i = 0; i < 10; i++)
  {
    if (atomic.compare_exchange_weak(expected, desired, success, failure) == true)
    {
      return true;
    }
  }

  return false;
}

template <class A, class T>
__host__ __device__ bool c_cmpxchg_weak_loop(A* atomic, T* expected, T desired)
{
  for (int i = 0; i < 10; i++)
  {
    if (cuda::std::atomic_compare_exchange_weak(atomic, expected, desired) == true)
    {
      return true;
    }
  }

  return false;
}

template <class A, class T>
__host__ __device__ bool
c_cmpxchg_weak_loop(A* atomic, T* expected, T desired, cuda::std::memory_order success, cuda::std::memory_order failure)
{
  for (int i = 0; i < 10; i++)
  {
    if (cuda::std::atomic_compare_exchange_weak_explicit(atomic, expected, desired, success, failure) == true)
    {
      return true;
    }
  }

  return false;
}
