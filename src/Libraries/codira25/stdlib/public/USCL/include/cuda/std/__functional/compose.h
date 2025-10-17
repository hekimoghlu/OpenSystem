/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2023-25 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_COMPOSE_H
#define _CUDA_STD___FUNCTIONAL_COMPOSE_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__functional/invoke.h>
#include <uscl/std/__functional/perfect_forward.h>
#include <uscl/std/__type_traits/decay.h>
#include <uscl/std/__utility/forward.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __compose_op
{
  template <class _Fn1, class _Fn2, class... _Args>
  _CCCL_API constexpr auto operator()(_Fn1&& __f1, _Fn2&& __f2, _Args&&... __args) const noexcept(noexcept(
    ::cuda::std::invoke(::cuda::std::forward<_Fn1>(__f1),
                        ::cuda::std::invoke(::cuda::std::forward<_Fn2>(__f2), ::cuda::std::forward<_Args>(__args)...))))
    -> decltype(::cuda::std::invoke(
      ::cuda::std::forward<_Fn1>(__f1),
      ::cuda::std::invoke(::cuda::std::forward<_Fn2>(__f2), ::cuda::std::forward<_Args>(__args)...)))
  {
    return ::cuda::std::invoke(
      ::cuda::std::forward<_Fn1>(__f1),
      ::cuda::std::invoke(::cuda::std::forward<_Fn2>(__f2), ::cuda::std::forward<_Args>(__args)...));
  }
};

template <class _Fn1, class _Fn2>
struct __compose_t : __perfect_forward<__compose_op, _Fn1, _Fn2>
{
  _LIBCUDACXX_DELEGATE_CONSTRUCTORS(__compose_t, __perfect_forward, __compose_op, _Fn1, _Fn2);
};

template <class _Fn1, class _Fn2>
_CCCL_API constexpr auto __compose(_Fn1&& __f1, _Fn2&& __f2) noexcept(noexcept(
  __compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(::cuda::std::forward<_Fn1>(__f1), ::cuda::std::forward<_Fn2>(__f2))))
  -> decltype(__compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(
    ::cuda::std::forward<_Fn1>(__f1), ::cuda::std::forward<_Fn2>(__f2)))
{
  return __compose_t<decay_t<_Fn1>, decay_t<_Fn2>>(::cuda::std::forward<_Fn1>(__f1), ::cuda::std::forward<_Fn2>(__f2));
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_COMPOSE_H
