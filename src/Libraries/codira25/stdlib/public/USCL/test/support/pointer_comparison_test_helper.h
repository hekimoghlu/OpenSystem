/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#ifndef POINTER_COMPARISON_TEST_HELPER_H
#define POINTER_COMPARISON_TEST_HELPER_H

#include <uscl/std/cassert>
#include <uscl/std/cstdint>

#include "test_macros.h"

template <template <class> class CompareTemplate>
__host__ __device__ void do_pointer_comparison_test()
{
  typedef CompareTemplate<int*> Compare;
  typedef CompareTemplate<cuda::std::uintptr_t> UIntCompare;
  typedef CompareTemplate<void> VoidCompare;

  Compare comp;
  UIntCompare ucomp;
  VoidCompare vcomp;
  struct
  {
    int a, b;
  } local;
  int* pointers[] = {&local.a, &local.b, nullptr, &local.a + 1};
  for (int* lhs : pointers)
  {
    for (int* rhs : pointers)
    {
      cuda::std::uintptr_t lhs_uint = reinterpret_cast<cuda::std::uintptr_t>(lhs);
      cuda::std::uintptr_t rhs_uint = reinterpret_cast<cuda::std::uintptr_t>(rhs);
      assert(comp(lhs, rhs) == ucomp(lhs_uint, rhs_uint));
      assert(vcomp(lhs, rhs) == ucomp(lhs_uint, rhs_uint));
    }
  }
}

template <class Comp>
__host__ __device__ void do_pointer_comparison_test(Comp comp)
{
  struct
  {
    int a, b;
  } local;
  int* pointers[] = {&local.a, &local.b, nullptr, &local.a + 1};
  for (int* lhs : pointers)
  {
    for (int* rhs : pointers)
    {
      cuda::std::uintptr_t lhs_uint = reinterpret_cast<cuda::std::uintptr_t>(lhs);
      cuda::std::uintptr_t rhs_uint = reinterpret_cast<cuda::std::uintptr_t>(rhs);
      void* lhs_void                = static_cast<void*>(lhs);
      void* rhs_void                = static_cast<void*>(rhs);
      assert(comp(lhs, rhs) == comp(lhs_uint, rhs_uint));
      assert(comp(lhs_void, rhs_void) == comp(lhs_uint, rhs_uint));
    }
  }
}

#endif // POINTER_COMPARISON_TEST_HELPER_H
