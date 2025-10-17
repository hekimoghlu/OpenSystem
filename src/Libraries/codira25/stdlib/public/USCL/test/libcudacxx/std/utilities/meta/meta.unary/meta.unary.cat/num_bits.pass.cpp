/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

#include <uscl/std/climits>
#include <uscl/std/complex>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class T, int Expected = sizeof(T) * CHAR_BIT>
__host__ __device__ void test_num_bits()
{
  static_assert(cuda::std::__num_bits_v<T> == Expected);
  static_assert(cuda::std::__num_bits_v<const T> == Expected);
  static_assert(cuda::std::__num_bits_v<volatile T> == Expected);
  static_assert(cuda::std::__num_bits_v<const volatile T> == Expected);
}

struct likely_padded
{
  char c;
  int i;
};

int main(int, char**)
{
  test_num_bits<char>();
  test_num_bits<short>();
  test_num_bits<int>();
  test_num_bits<long>();
  test_num_bits<long long>();
#if _CCCL_HAS_INT128()
  test_num_bits<__uint128_t>();
  test_num_bits<__int128_t>();
#endif // _CCCL_HAS_INT128()
  test_num_bits<float>();
  test_num_bits<double>();
  test_num_bits<cuda::std::complex<float>>();
  test_num_bits<cuda::std::complex<double>>();
#if _CCCL_HAS_NVBF16()
  test_num_bits<__half>();
  test_num_bits<__half2>();
  test_num_bits<cuda::std::complex<__half>>();
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP16()
  test_num_bits<__nv_bfloat16>();
  test_num_bits<__nv_bfloat162>();
  test_num_bits<cuda::std::complex<__nv_bfloat16>>();
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_FLOAT128()
  test_num_bits<__float128>();
  test_num_bits<cuda::std::complex<__float128>>();
#endif // _CCCL_HAS_FLOAT128()
#if _CCCL_HAS_NVFP8_E4M3()
  test_num_bits<__nv_fp8_e4m3>();
#endif
#if _CCCL_HAS_NVFP8_E5M2()
  test_num_bits<__nv_fp8_e5m2>();
#endif
#if _CCCL_HAS_NVFP8_E8M0()
  test_num_bits<__nv_fp8_e8m0>();
#endif
#if _CCCL_HAS_NVFP6_E3M2()
  test_num_bits<__nv_fp6_e3m2, 6>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP6_E2M3()
  test_num_bits<__nv_fp6_e2m3, 6>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_num_bits<__nv_fp4_e2m1, 4>();
#endif // _CCCL_HAS_NVFP8_E8M0()
  test_num_bits<int*>();
  return 0;
}
