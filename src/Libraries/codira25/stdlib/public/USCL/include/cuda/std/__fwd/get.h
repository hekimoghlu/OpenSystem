/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FWD_GET_H
#define _CUDA_STD___FWD_GET_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__concepts/copyable.h>
#include <uscl/std/__fwd/array.h>
#include <uscl/std/__fwd/complex.h>
#include <uscl/std/__fwd/pair.h>
#include <uscl/std/__fwd/subrange.h>
#include <uscl/std/__fwd/tuple.h>
#include <uscl/std/__tuple_dir/tuple_element.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>& get(tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>& get(const tuple<_Tp...>&) noexcept;

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr tuple_element_t<_Ip, tuple<_Tp...>>&& get(tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class... _Tp>
_CCCL_API constexpr const tuple_element_t<_Ip, tuple<_Tp...>>&& get(const tuple<_Tp...>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>& get(pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>& get(const pair<_T1, _T2>&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr tuple_element_t<_Ip, pair<_T1, _T2>>&& get(pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _T1, class _T2>
_CCCL_API constexpr const tuple_element_t<_Ip, pair<_T1, _T2>>&& get(const pair<_T1, _T2>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr _Tp& get(array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr const _Tp& get(const array<_Tp, _Size>&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr _Tp&& get(array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp, size_t _Size>
_CCCL_API constexpr const _Tp&& get(const array<_Tp, _Size>&&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr _Tp& get(complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr _Tp&& get(complex<_Tp>&&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr const _Tp& get(const complex<_Tp>&) noexcept;

template <size_t _Ip, class _Tp>
_CCCL_API constexpr const _Tp&& get(const complex<_Tp>&&) noexcept;

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_RANGES

#if _CCCL_HAS_CONCEPTS()
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires((_Index == 0) && copyable<_Iter>) || (_Index == 1)
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <size_t _Index,
          class _Iter,
          class _Sent,
          subrange_kind _Kind,
          enable_if_t<((_Index == 0) && copyable<_Iter>) || (_Index == 1), int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
_CCCL_API constexpr auto get(const subrange<_Iter, _Sent, _Kind>& __subrange);

#if _CCCL_HAS_CONCEPTS()
template <size_t _Index, class _Iter, class _Sent, subrange_kind _Kind>
  requires(_Index < 2)
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <
  size_t _Index,
  class _Iter,
  class _Sent,
  subrange_kind _Kind,
  enable_if_t<_Index<2, int> = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
_CCCL_API constexpr auto get(subrange<_Iter, _Sent, _Kind>&& __subrange);

_CCCL_END_NAMESPACE_RANGES

_CCCL_BEGIN_NAMESPACE_CUDA_STD

using ::cuda::std::ranges::get;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FWD_GET_H
