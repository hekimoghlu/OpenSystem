/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

// #include <uscl/std/functional>
#include <uscl/std/utility>

#include "test_macros.h"

struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
typedef Holder<Incomplete>* Ptr;

__host__ __device__ Ptr no_args()
{
  return nullptr;
}
__host__ __device__ Ptr one_arg(Ptr p)
{
  return p;
}
__host__ __device__ Ptr two_args(Ptr p, Ptr)
{
  return p;
}
__host__ __device__ Ptr three_args(Ptr p, Ptr, Ptr)
{
  return p;
}

__host__ __device__ void one_arg_void(Ptr) {}

int main(int, char**)
{
  Ptr x        = nullptr;
  const Ptr cx = nullptr;
  cuda::std::ref(no_args)();
  cuda::std::ref(one_arg)(x);
  cuda::std::ref(one_arg)(cx);
  cuda::std::ref(two_args)(x, x);
  cuda::std::ref(two_args)(x, cx);
  cuda::std::ref(two_args)(cx, x);
  cuda::std::ref(two_args)(cx, cx);
  cuda::std::ref(three_args)(x, x, x);
  cuda::std::ref(three_args)(x, x, cx);
  cuda::std::ref(three_args)(x, cx, x);
  cuda::std::ref(three_args)(cx, x, x);
  cuda::std::ref(three_args)(x, cx, cx);
  cuda::std::ref(three_args)(cx, x, cx);
  cuda::std::ref(three_args)(cx, cx, x);
  cuda::std::ref(three_args)(cx, cx, cx);
  cuda::std::ref(one_arg_void)(x);
  cuda::std::ref(one_arg_void)(cx);

  return 0;
}
