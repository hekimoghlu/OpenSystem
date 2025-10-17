/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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

#ifndef _CUDA_STD___TUPLE_MAKE_TUPLE_TYPES_H
#define _CUDA_STD___TUPLE_MAKE_TUPLE_TYPES_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__fwd/array.h>
#include <uscl/std/__fwd/tuple.h>
#include <uscl/std/__tuple_dir/tuple_element.h>
#include <uscl/std/__tuple_dir/tuple_indices.h>
#include <uscl/std/__tuple_dir/tuple_size.h>
#include <uscl/std/__tuple_dir/tuple_types.h>
#include <uscl/std/__type_traits/copy_cvref.h>
#include <uscl/std/__type_traits/remove_cv.h>
#include <uscl/std/__type_traits/remove_reference.h>
#include <uscl/std/__type_traits/type_list.h>
#include <uscl/std/cstddef>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// __make_tuple_types<_Tuple<_Types...>, _Ep, _Sp>::type is a
// __tuple_types<_Types...> using only those _Types in the range [_Sp, _Ep).
// _Sp defaults to 0 and _Ep defaults to tuple_size<_Tuple>.  If _Tuple is a
// lvalue_reference type, then __tuple_types<_Types&...> is the result.

template <class _TupleTypes, class _TupleIndices>
struct __make_tuple_types_flat;

template <template <class...> class _Tuple, class... _Types, size_t... _Idx>
struct __make_tuple_types_flat<_Tuple<_Types...>, __tuple_indices<_Idx...>>
{
  using __tuple_types_list = __type_list<_Types...>;

  // Specialization for pair, tuple, and __tuple_types
  template <class _Tp, class _ApplyFn = __apply_cvref_fn<_Tp>>
  using __apply_quals _CCCL_NODEBUG_ALIAS =
    __tuple_types<__type_call<_ApplyFn, __type_at_c<_Idx, __tuple_types_list>>...>;
};

template <class _Vt, size_t _Np, size_t... _Idx>
struct __make_tuple_types_flat<array<_Vt, _Np>, __tuple_indices<_Idx...>>
{
  template <size_t>
  using __value_type = _Vt;
  template <class _Tp, class _ApplyFn = __apply_cvref_fn<_Tp>>
  using __apply_quals = __tuple_types<__type_call<_ApplyFn, __value_type<_Idx>>...>;
};

template <class _Tp,
          size_t _Ep     = tuple_size<remove_reference_t<_Tp>>::value,
          size_t _Sp     = 0,
          bool _SameSize = (_Ep == tuple_size<remove_reference_t<_Tp>>::value)>
struct __make_tuple_types
{
  static_assert(_Sp <= _Ep, "__make_tuple_types input error");
  using _RawTp = remove_cv_t<remove_reference_t<_Tp>>;
  using _Maker = __make_tuple_types_flat<_RawTp, __make_tuple_indices_t<_Ep, _Sp>>;
  using type   = typename _Maker::template __apply_quals<_Tp>;
};

template <class... _Types, size_t _Ep>
struct __make_tuple_types<tuple<_Types...>, _Ep, 0, true>
{
  using type _CCCL_NODEBUG_ALIAS = __tuple_types<_Types...>;
};

template <class... _Types, size_t _Ep>
struct __make_tuple_types<__tuple_types<_Types...>, _Ep, 0, true>
{
  using type _CCCL_NODEBUG_ALIAS = __tuple_types<_Types...>;
};

template <class _Tp, size_t _Ep = tuple_size<remove_reference_t<_Tp>>::value, size_t _Sp = 0>
using __make_tuple_types_t = typename __make_tuple_types<_Tp, _Ep, _Sp>::type;

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_MAKE_TUPLE_TYPES_H
