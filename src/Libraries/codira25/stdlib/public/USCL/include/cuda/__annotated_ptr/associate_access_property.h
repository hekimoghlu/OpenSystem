/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#ifndef _CUDA___ANNOTATED_PTR_ASSOCIATE_ACCESS_PROPERTY_H
#define _CUDA___ANNOTATED_PTR_ASSOCIATE_ACCESS_PROPERTY_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/__annotated_ptr/access_property.h>
#include <uscl/__memory/address_space.h>
#include <uscl/std/__type_traits/always_false.h>
#include <uscl/std/__type_traits/is_one_of.h>
#include <uscl/std/__type_traits/is_same.h>
#include <uscl/std/cstdint>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//----------------------------------------------------------------------------------------------------------------------
// Private access property methods

template <typename _Property>
inline constexpr bool __is_access_property_v =
  ::cuda::std::__is_one_of_v<_Property,
                             access_property::shared,
                             access_property::global,
                             access_property::normal,
                             access_property::persisting,
                             access_property::streaming,
                             access_property>;

template <typename _Property>
inline constexpr bool __is_global_access_property_v =
  ::cuda::std::__is_one_of_v<_Property,
                             access_property::global,
                             access_property::normal,
                             access_property::persisting,
                             access_property::streaming,
                             access_property>;

#if _CCCL_CUDA_COMPILATION()

template <typename _Property>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void*
__associate_address_space(void* __ptr, [[maybe_unused]] _Property __prop)
{
  if constexpr (::cuda::std::is_same_v<_Property, access_property::shared>)
  {
    [[maybe_unused]] bool __b = ::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::shared);
    _CCCL_ASSERT(__b, "");
    _CCCL_ASSUME(__b);
  }
  else if constexpr (__is_global_access_property_v<_Property>)
  {
    [[maybe_unused]] bool __b = ::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::global);
    _CCCL_ASSERT(__b, "");
    _CCCL_ASSUME(__b);
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Property>, "invalid access_property");
  }
  return __ptr;
}

_CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __associate_raw_descriptor(void* __ptr, [[maybe_unused]] uint64_t __prop)
{
  NV_IF_TARGET(NV_PROVIDES_SM_80, (return ::__nv_associate_access_property(__ptr, __prop);))
  return __ptr;
}

template <typename _Property>
[[nodiscard]] _CCCL_HIDE_FROM_ABI _CCCL_DEVICE void* __associate_descriptor(void* __ptr, _Property __prop)
{
  static_assert(__is_access_property_v<_Property>, "invalid cuda::access_property");
  if constexpr (!::cuda::std::is_same_v<_Property, access_property::shared>)
  {
    [[maybe_unused]] auto __raw_prop = static_cast<uint64_t>(access_property{__prop});
    return ::cuda::__associate_raw_descriptor(__ptr, __raw_prop);
  }
  return __ptr;
}

#endif // _CCCL_CUDA_COMPILATION()

template <typename _Type, typename _Property>
[[nodiscard]] _CCCL_API inline _Type* __associate(_Type* __ptr, [[maybe_unused]] _Property __prop) noexcept
{
  static_assert(__is_access_property_v<_Property>, "invalid cuda::access_property");
  NV_IF_ELSE_TARGET(
    NV_IS_DEVICE,
    (auto __void_ptr       = const_cast<void*>(static_cast<const void*>(__ptr));
     auto __associated_ptr = ::cuda::__associate_address_space(__void_ptr, __prop);
     return static_cast<_Type*>(::cuda::__associate_descriptor(__associated_ptr, __prop));),
    (return __ptr;))
}

//----------------------------------------------------------------------------------------------------------------------
// Public access property methods

template <typename _Tp, typename _Property>
[[nodiscard]] _CCCL_API inline _Tp* associate_access_property(_Tp* __ptr, _Property __prop) noexcept
{
  static_assert(__is_access_property_v<_Property>, "invalid cuda::access_property");
  return ::cuda::__associate(__ptr, __prop);
}

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA___ANNOTATED_PTR_ASSOCIATE_ACCESS_PROPERTY_H
