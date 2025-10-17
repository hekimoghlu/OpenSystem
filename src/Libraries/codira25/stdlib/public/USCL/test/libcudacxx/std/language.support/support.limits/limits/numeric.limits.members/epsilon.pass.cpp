/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 24, 2024.
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

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

// test numeric_limits

// epsilon()

#include <uscl/std/bit>
#include <uscl/std/cassert>
#include <uscl/std/cfloat>
#include <uscl/std/limits>

#include "common.h"
#include "test_macros.h"

template <class T>
__host__ __device__ void test(T expected)
{
  assert(float_eq(cuda::std::numeric_limits<T>::epsilon(), expected));
  assert(float_eq(cuda::std::numeric_limits<const T>::epsilon(), expected));
  assert(float_eq(cuda::std::numeric_limits<volatile T>::epsilon(), expected));
  assert(float_eq(cuda::std::numeric_limits<const volatile T>::epsilon(), expected));
}

int main(int, char**)
{
  test<bool>(false);
  test<char>(0);
  test<signed char>(0);
  test<unsigned char>(0);
  test<wchar_t>(0);
#if TEST_STD_VER > 2017 && defined(__cpp_char8_t)
  test<char8_t>(0);
#endif
  test<char16_t>(0);
  test<char32_t>(0);
  test<short>(0);
  test<unsigned short>(0);
  test<int>(0);
  test<unsigned int>(0);
  test<long>(0);
  test<unsigned long>(0);
  test<long long>(0);
  test<unsigned long long>(0);
#if _CCCL_HAS_INT128()
  test<__int128_t>(0);
  test<__uint128_t>(0);
#endif // _CCCL_HAS_INT128()
  test<float>(FLT_EPSILON);
  test<double>(DBL_EPSILON);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(LDBL_EPSILON);
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test<__half>(__double2half(0.0009765625));
#endif // _CCCL_HAS_NVFP16
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16>(__double2bfloat16(0.0078125));
#endif // _CCCL_HAS_NVBF16
#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>(make_fp8_e4m3(0.125));
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>(make_fp8_e5m2(0.25));
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>(make_fp8_e8m0(1.0));
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>(make_fp6_e2m3(0.125));
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>(make_fp6_e3m2(0.25));
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>(make_fp4_e2m1(0.5));
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test<__float128>(cuda::std::bit_cast<__float128>(__uint128_t{0x3f8f000000000000} << 64));
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
