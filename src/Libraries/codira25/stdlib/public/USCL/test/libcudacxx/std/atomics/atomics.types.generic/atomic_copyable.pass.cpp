/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 6, 2022.
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

#include <uscl/atomic>
#include <uscl/std/atomic>
#include <uscl/std/cassert>
#include <uscl/std/utility>
// #include <uscl/std/thread> // for thread_id
// #include <uscl/std/chrono> // for nanoseconds

#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test_not_copy_constructible()
{
  static_assert(!cuda::std::is_constructible<T, T&&>(), "");
  static_assert(!cuda::std::is_constructible<T, const T&>(), "");
  static_assert(!cuda::std::is_assignable<T, T&&>(), "");
  static_assert(!cuda::std::is_assignable<T, const T&>(), "");
}

template <class T>
__host__ __device__ void test_copy_constructible()
{
  static_assert(cuda::std::is_constructible<T, T&&>(), "");
  static_assert(cuda::std::is_constructible<T, const T&>(), "");
  static_assert(!cuda::std::is_assignable<T, T&&>(), "");
  static_assert(!cuda::std::is_assignable<T, const T&>(), "");
}

template <class T, class A>
__host__ __device__ void test_atomic_ref_copy_ctor()
{
  SHARED A val;
  val = 0;

  T t0(val);
  T t1(t0);

  t0++;
  t1++;

  assert(t1.load() == 2);
}

template <class T, class A>
__host__ __device__ void test_atomic_ref_move_ctor()
{
  SHARED A val;
  val = 0;

  T t0(val);
  t0++;

  T t1(cuda::std::move(t0));
  t1++;

  assert(t1.load() == 2);
}

int main(int, char**)
{
  test_not_copy_constructible<cuda::std::atomic<int>>();
  test_not_copy_constructible<cuda::atomic<int>>();

  test_copy_constructible<cuda::std::atomic_ref<int>>();
  test_copy_constructible<cuda::atomic_ref<int>>();

  test_atomic_ref_copy_ctor<cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_copy_ctor<cuda::atomic_ref<int>, int>();
  test_atomic_ref_copy_ctor<const cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_copy_ctor<const cuda::atomic_ref<int>, int>();

  test_atomic_ref_move_ctor<cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_move_ctor<cuda::atomic_ref<int>, int>();
  test_atomic_ref_move_ctor<const cuda::std::atomic_ref<int>, int>();
  test_atomic_ref_move_ctor<const cuda::atomic_ref<int>, int>();
  // test(cuda::std::this_thread::get_id());
  // test(cuda::std::chrono::nanoseconds(2));

  return 0;
}
