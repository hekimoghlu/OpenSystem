/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#ifndef _CUDA___UTILITY_BASIC_ANY_VIRTUAL_TABLES_H
#define _CUDA___UTILITY_BASIC_ANY_VIRTUAL_TABLES_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/__utility/__basic_any/basic_any_fwd.h>
#include <uscl/__utility/__basic_any/interfaces.h>
#include <uscl/__utility/__basic_any/rtti.h>
#include <uscl/__utility/__basic_any/virtual_functions.h>
#include <uscl/__utility/__basic_any/virtual_ptrs.h>
#include <uscl/std/__exception/terminate.h>
#include <uscl/std/__utility/typeid.h>

#include <nv/target>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Interface>
using __vtable_for _CCCL_NODEBUG_ALIAS = typename __overrides_for_t<_Interface>::__vtable;

//!
//! __basic_vtable
//!
template <class _Interface, auto... _Mbrs>
struct _CCCL_DECLSPEC_EMPTY_BASES __basic_vtable
    : __rtti_base
    , __virtual_fn<_Mbrs>...
{
  using interface _CCCL_NODEBUG_ALIAS = _Interface;
  static constexpr size_t __cbases    = ::cuda::std::__type_list_size<__unique_interfaces<interface>>::value;

  template <class _VPtr, class _Tp, class... _OtherMembers, class... _Interfaces>
  _CCCL_API constexpr __basic_vtable(
    _VPtr __vptr, __overrides_list<_Tp, _OtherMembers...>, __tag<_Interfaces...>) noexcept
      : __rtti_base{__vtable_kind::__normal, __cbases, _CCCL_TYPEID(__basic_vtable)}
      , __virtual_fn<_Mbrs>{__override_tag<_Tp, _OtherMembers::value>{}}...
      , __vptr_map_{__base_vptr{__vptr->__query_interface(_Interfaces())}...}
  {}

  template <class _Tp, class _VPtr>
  _CCCL_API constexpr __basic_vtable(__tag<_Tp>, _VPtr __vptr) noexcept
      : __basic_vtable{__vptr,
                       __overrides_for_t<interface, _Tp>(),
                       __unique_interfaces<interface, ::cuda::std::__type_quote<__tag>>()}
  {}

  [[nodiscard]] _CCCL_API auto __query_interface(interface) const noexcept -> __vptr_for<interface>
  {
    return this;
  }

  template <class... _Others>
  [[nodiscard]] _CCCL_API auto __query_interface(__iset_<_Others...>) const noexcept -> __vptr_for<__iset_<_Others...>>
  {
    using __remainder _CCCL_NODEBUG_ALIAS =
      ::cuda::std::__type_list_size<::cuda::std::__type_find<__unique_interfaces<interface>, __iset_<_Others...>>>;
    constexpr size_t __index = __cbases - __remainder::value;
    if constexpr (__index < __cbases)
    {
      // `_Interface` extends __iset_<_Others...> exactly. We can return an actual
      // vtable pointer.
      return static_cast<__vtable_for<__iset_<_Others...>> const*>(__vptr_map_[__index]);
    }
    else
    {
      // Otherwise, we have to return a subset vtable pointer, which does
      // dynamic interface lookup.
      return static_cast<__vptr_for<__iset_<_Others...>>>(__query_interface(__iunknown()));
    }
  }

  template <class _Other>
  [[nodiscard]] _CCCL_API auto __query_interface(_Other) const noexcept -> __vptr_for<_Other>
  {
    constexpr size_t __index = __index_of_base<_Other, interface>::value;
    static_assert(__index < __cbases);
    return static_cast<__vptr_for<_Other>>(__vptr_map_[__index]);
  }

  __base_vptr __vptr_map_[__cbases];
};

//!
//! __vtable implementation details
//!

template <class... _Interfaces>
struct _CCCL_DECLSPEC_EMPTY_BASES __vtable_tuple
    : __rtti_ex<sizeof...(_Interfaces)>
    , __vtable_for<_Interfaces>...
{
  static_assert((::cuda::std::is_class_v<_Interfaces> && ...), "expected class types");

  template <class _Tp, class _Super>
  _CCCL_API constexpr __vtable_tuple(__tag<_Tp, _Super> __type) noexcept
      : __rtti_ex<sizeof...(_Interfaces)>{__type, __tag<_Interfaces...>(), this}
#if _CCCL_COMPILER(MSVC)
      // workaround for MSVC bug
      , __overrides_for_t<_Interfaces>::__vtable{__tag<_Tp>(), this}...
#else // ^^^ MSVC ^^^ / vvv !MSVC vvv
      , __vtable_for<_Interfaces>{__tag<_Tp>(), this}...
#endif // !MSVC
  {
    static_assert(::cuda::std::is_class_v<_Super>, "expected a class type");
  }

  _CCCL_TEMPLATE(class _Interface)
  _CCCL_REQUIRES(::cuda::std::__is_included_in_v<_Interface, _Interfaces...>)
  [[nodiscard]] _CCCL_API constexpr auto __query_interface(_Interface) const noexcept -> __vptr_for<_Interface>
  {
    return static_cast<__vptr_for<_Interface>>(this);
  }
};

// The vtable type for type `_Interface` is a `__vtable_tuple` of `_Interface`
// and all of its base interfaces.
template <class _Interface>
using __vtable _CCCL_NODEBUG_ALIAS = __unique_interfaces<_Interface, ::cuda::std::__type_quote<__vtable_tuple>>;

// __vtable_for_v<_Interface, _Tp> is an instance of `__vtable<_Interface>` that
// contains the overrides for `_Tp`.
template <class _Interface, class _Tp>
_CCCL_GLOBAL_CONSTANT __vtable<_Interface> __vtable_for_v{__tag<_Tp, _Interface>()};

template <class _Interface, class _Tp>
_CCCL_API constexpr __vtable<_Interface> const* __get_vtable_ptr_for() noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return &__vtable_for_v<_Interface, _Tp>;), (::cuda::std::terminate();))
}

_CCCL_END_NAMESPACE_CUDA

#include <uscl/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_VIRTUAL_TABLES_H
