/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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

// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___MDSPAN_DEFAULT_ACCESSOR_H
#define _CUDA_STD___MDSPAN_DEFAULT_ACCESSOR_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__fwd/mdspan.h>
#include <uscl/std/__type_traits/is_abstract.h>
#include <uscl/std/__type_traits/is_array.h>
#include <uscl/std/__type_traits/is_convertible.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _ElementType>
struct default_accessor
{
  static_assert(!is_array_v<_ElementType>, "default_accessor: template argument may not be an array type");
  static_assert(!is_abstract_v<_ElementType>, "default_accessor: template argument may not be an abstract class");

  using offset_policy    = default_accessor;
  using element_type     = _ElementType;
  using reference        = _ElementType&;
  using data_handle_type = _ElementType*;

  _CCCL_HIDE_FROM_ABI constexpr default_accessor() noexcept = default;

  _CCCL_TEMPLATE(class _OtherElementType)
  _CCCL_REQUIRES(is_convertible_v<_OtherElementType (*)[], element_type (*)[]>)
  _CCCL_API constexpr default_accessor(default_accessor<_OtherElementType>) noexcept {}

  [[nodiscard]] _CCCL_API constexpr reference access(data_handle_type __p, size_t __i) const noexcept
  {
    return __p[__i];
  }
  [[nodiscard]] _CCCL_API constexpr data_handle_type offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __p + __i;
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MDSPAN_DEFAULT_ACCESSOR_H
