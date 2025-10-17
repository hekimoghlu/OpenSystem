/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H
#define _CUDA_STD___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__floating_point/cuda_fp_types.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
inline constexpr bool __is_extended_floating_point_v = false;

#if _CCCL_HAS_NVFP16()
template <>
inline constexpr bool __is_extended_floating_point_v<__half> = true;
#endif // _CCCL_HAS_NVFP16

#if _CCCL_HAS_NVBF16()
template <>
inline constexpr bool __is_extended_floating_point_v<__nv_bfloat16> = true;
#endif // _CCCL_HAS_NVBF16

#if _CCCL_HAS_NVFP8_E4M3()
template <>
inline constexpr bool __is_extended_floating_point_v<__nv_fp8_e4m3> = true;
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
template <>
inline constexpr bool __is_extended_floating_point_v<__nv_fp8_e5m2> = true;
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
template <>
inline constexpr bool __is_extended_floating_point_v<__nv_fp8_e8m0> = true;
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
template <>
inline constexpr bool __is_extended_floating_point_v<__nv_fp6_e2m3> = true;
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
template <>
inline constexpr bool __is_extended_floating_point_v<__nv_fp6_e3m2> = true;
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
template <>
inline constexpr bool __is_extended_floating_point_v<__nv_fp4_e2m1> = true;
#endif // _CCCL_HAS_NVFP4_E2M1()

#if _CCCL_HAS_FLOAT128()
template <>
inline constexpr bool __is_extended_floating_point_v<__float128> = true;
#endif // _CCCL_HAS_FLOAT128()

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TYPE_TRAITS_IS_EXTENDED_FLOATING_POINT_H
