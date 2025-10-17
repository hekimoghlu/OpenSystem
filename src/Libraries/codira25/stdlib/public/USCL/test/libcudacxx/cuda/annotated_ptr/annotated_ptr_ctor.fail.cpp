/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "utils.h"

template <typename T, typename P>
__host__ __device__ __noinline__ void test_ctor()
{
  // default ctor, cpy and cpy assignment
  cuda::annotated_ptr<T, P> def;
  def = def;
  cuda::annotated_ptr<T, P> other(def);

  // from ptr
  T* rp = nullptr;
  cuda::annotated_ptr<T, P> a(rp);
  assert(!a);

  // cpy ctor & assign to cv
  cuda::annotated_ptr<const T, P> c(def);
  cuda::annotated_ptr<volatile T, P> d(def);
  cuda::annotated_ptr<const volatile T, P> e(def);
  c = e; // FAIL
  d = d; // FAIL
}

template <typename T, typename P>
__host__ __device__ __noinline__ void test_global_ctor()
{
  test_ctor<T, P>();
}

__host__ __device__ __noinline__ void test_global_ctors()
{
  test_global_ctor<int, cuda::access_property::normal>();
}

int main(int argc, char** argv)
{
  test_global_ctors();
  return 0;
}
