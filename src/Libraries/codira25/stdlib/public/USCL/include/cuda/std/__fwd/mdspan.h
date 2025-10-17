/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 22, 2022.
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

#ifndef _CUDA_STD___FWD_MDSPAN_H
#define _CUDA_STD___FWD_MDSPAN_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/void_t.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// [mdspan.accessor.default]
template <class _ElementType>
struct default_accessor;

// Layout policy with a mapping which corresponds to Fortran-style array layouts
struct layout_left
{
  template <class _Extents>
  class mapping;
};

// Layout policy with a mapping which corresponds to C-style array layouts
struct layout_right
{
  template <class _Extents>
  class mapping;
};

// Layout policy with a unique mapping where strides are arbitrary
struct layout_stride
{
  template <class Extents>
  class mapping;
};

// [mdspan.layout.policy.reqmts]
namespace __mdspan_detail
{
template <class _Layout, class _Extents, class = void>
inline constexpr bool __is_valid_layout_mapping = false;

template <class _Layout, class _Extents>
inline constexpr bool
  __is_valid_layout_mapping<_Layout, _Extents, void_t<typename _Layout::template mapping<_Extents>>> = true;
} // namespace __mdspan_detail

// [mdspan.mdspan]
template <class _ElementType,
          class _Extents,
          class _LayoutPolicy   = layout_right,
          class _AccessorPolicy = default_accessor<_ElementType>>
class mdspan;

template <class _Tp>
inline constexpr bool __is_std_mdspan_v = false;

template <class _ElementType, class _Extents, class _LayoutPolicy, class _AccessorPolicy>
inline constexpr bool __is_std_mdspan_v<mdspan<_ElementType, _Extents, _LayoutPolicy, _AccessorPolicy>> = true;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_MDSPAN_H
