/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
#ifndef _CUDA_FUNCTIONAL_MINIMUM_H
#define _CUDA_FUNCTIONAL_MINIMUM_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/std/__type_traits/common_type.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minimum
{
  _CCCL_EXEC_CHECK_DISABLE
  [[nodiscard]] _CCCL_API constexpr _Tp operator()(const _Tp& __lhs, const _Tp& __rhs) const
    noexcept(noexcept((__lhs < __rhs) ? __lhs : __rhs))
  {
    return (__lhs < __rhs) ? __lhs : __rhs;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(minimum);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minimum<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr ::cuda::std::common_type_t<_T1, _T2>
  operator()(const _T1& __lhs, const _T2& __rhs) const noexcept(noexcept((__lhs < __rhs) ? __lhs : __rhs))
  {
    return (__lhs < __rhs) ? __lhs : __rhs;
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA_FUNCTIONAL_MINIMUM_H
