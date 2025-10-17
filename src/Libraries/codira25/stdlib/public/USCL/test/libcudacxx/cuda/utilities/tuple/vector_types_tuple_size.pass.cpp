/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 5, 2024.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "test_macros.h"

_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class VType, size_t Size>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::tuple_size<VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<volatile VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const volatile VType>::value == Size, "");
}

#define EXPAND_VECTOR_TYPE(Type) \
  test<Type##1, 1>();            \
  test<Type##2, 2>();            \
  test<Type##3, 3>();            \
  test<Type##4, 4>();

__host__ __device__ constexpr bool test()
{
  EXPAND_VECTOR_TYPE(char);
  EXPAND_VECTOR_TYPE(uchar);
  EXPAND_VECTOR_TYPE(short);
  EXPAND_VECTOR_TYPE(ushort);
  EXPAND_VECTOR_TYPE(int);
  EXPAND_VECTOR_TYPE(uint);
  EXPAND_VECTOR_TYPE(long);
  EXPAND_VECTOR_TYPE(ulong);
  EXPAND_VECTOR_TYPE(longlong);
  EXPAND_VECTOR_TYPE(ulonglong);
  EXPAND_VECTOR_TYPE(float);
  EXPAND_VECTOR_TYPE(double);

#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, 4>();
  test<long4_32a, 4>();
  test<ulong4_16a, 4>();
  test<ulong4_32a, 4>();
  test<longlong4_16a, 4>();
  test<longlong4_32a, 4>();
  test<ulonglong4_16a, 4>();
  test<ulonglong4_32a, 4>();
  test<double4_16a, 4>();
  test<double4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

#if _CCCL_HAS_NVFP16()
  test<__half2, 2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat162, 2>();
#endif // _CCCL_HAS_NVBF16()

  test<dim3, 3>();

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
