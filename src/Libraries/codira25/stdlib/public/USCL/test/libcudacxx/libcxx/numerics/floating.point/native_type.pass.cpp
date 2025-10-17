/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

//// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <uscl/std/__floating_point/fp.h>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <cuda::std::__fp_format format, class T>
__host__ __device__ void test_fp_native_type()
{
  static_assert(cuda::std::is_same_v<T, cuda::std::__fp_native_type_t<format>>);
  static_assert(cuda::std::is_void_v<T> == !cuda::std::__fp_has_native_type_v<format>);
}

int main(int, char**)
{
  test_fp_native_type<cuda::std::__fp_format::__binary16, void>();
  test_fp_native_type<cuda::std::__fp_format::__binary32, float>();
  test_fp_native_type<cuda::std::__fp_format::__binary64, double>();
#if _CCCL_HAS_FLOAT128()
  test_fp_native_type<cuda::std::__fp_format::__binary128, __float128>();
#elif _CCCL_HAS_LONG_DOUBLE() && LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 113
  test_fp_native_type<cuda::std::__fp_format::__binary128, long double>();
#else // ^^^ has native binary128 ^^^ / vvv no native binary128 vvv
  test_fp_native_type<cuda::std::__fp_format::__binary128, void>();
#endif // ^^^ no native binary128 ^^^
  test_fp_native_type<cuda::std::__fp_format::__bfloat16, void>();
#if _CCCL_HAS_LONG_DOUBLE() && LDBL_MIN_EXP == -16381 && LDBL_MAX_EXP == 16384 && LDBL_MANT_DIG == 64
  test_fp_native_type<cuda::std::__fp_format::__fp80_x86, long double>();
#else // ^^^ has native __fp80_x86 ^^^ / vvv no native __fp80_x86 vvv
  test_fp_native_type<cuda::std::__fp_format::__fp80_x86, void>();
#endif // ^^^ no native __fp80_x86 ^^^
  test_fp_native_type<cuda::std::__fp_format::__fp8_nv_e4m3, void>();
  test_fp_native_type<cuda::std::__fp_format::__fp8_nv_e5m2, void>();
  test_fp_native_type<cuda::std::__fp_format::__fp8_nv_e8m0, void>();
  test_fp_native_type<cuda::std::__fp_format::__fp6_nv_e2m3, void>();
  test_fp_native_type<cuda::std::__fp_format::__fp6_nv_e3m2, void>();
  test_fp_native_type<cuda::std::__fp_format::__fp4_nv_e2m1, void>();

  return 0;
}
