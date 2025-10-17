/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 28, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA__TYPE_TRAITS_IS_FLOATING_POINT_H
#define __CUDA__TYPE_TRAITS_IS_FLOATING_POINT_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/integral_constant.h>
#include <uscl/std/__type_traits/is_extended_floating_point.h>
#include <uscl/std/__type_traits/is_floating_point.h>
#include <uscl/std/__type_traits/remove_cv.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! Tells whether a type is a floating point type, including extended floating point types.
//! Users are allowed to specialize this template for their own types.
template <class _Tp>
inline constexpr bool is_floating_point_v =
  ::cuda::std::is_floating_point_v<::cuda::std::remove_cv_t<_Tp>>
  || ::cuda::std::__is_extended_floating_point_v<::cuda::std::remove_cv_t<_Tp>>;

// we define the trait as alias, so users cannot specialize it (they should specialize the variable template instead)
template <class _Tp>
using is_floating_point = ::cuda::std::bool_constant<is_floating_point_v<_Tp>>;

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // __CUDA__TYPE_TRAITS_IS_FLOATING_POINT_H
