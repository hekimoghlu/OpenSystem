/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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

// <cuda/std/complex>

//   template<size_t I, class T> struct tuple_element;

#include <uscl/std/cassert>
#include <uscl/std/complex>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <size_t I, typename C, typename = void>
struct HasTupleElement : cuda::std::false_type
{};

template <size_t I, typename C>
struct HasTupleElement<I, C, cuda::std::void_t<decltype(cuda::std::tuple_element<I, C>{})>> : cuda::std::true_type
{};

struct SomeObject
{};

static_assert(!HasTupleElement<0, SomeObject>::value, "");
static_assert(!HasTupleElement<1, SomeObject>::value, "");
static_assert(!HasTupleElement<3, SomeObject>::value, "");

template <typename T>
__host__ __device__ void test()
{
  using C = cuda::std::complex<T>;

  static_assert(HasTupleElement<0, C>::value, "");
  static_assert(HasTupleElement<1, C>::value, "");

  static_assert(cuda::std::is_same_v<T, typename cuda::std::tuple_element<0, C>::type>);
  static_assert(cuda::std::is_same_v<T, typename cuda::std::tuple_element<1, C>::type>);
}

__host__ __device__ void test()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int, char**)
{
  return 0;
}
