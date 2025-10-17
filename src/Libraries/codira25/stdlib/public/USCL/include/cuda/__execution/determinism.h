/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 2, 2025.
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
#ifndef __CUDA___EXECUTION_DETERMINISM_H
#define __CUDA___EXECUTION_DETERMINISM_H

#include <uscl/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <uscl/__execution/require.h>
#include <uscl/std/__concepts/concept_macros.h>
#include <uscl/std/__execution/env.h>
#include <uscl/std/__type_traits/is_one_of.h>

#include <uscl/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

namespace determinism
{

struct __get_determinism_t;

enum class __determinism_t
{
  __not_guaranteed,
  __run_to_run,
  __gpu_to_gpu
};

template <__determinism_t _Guarantee>
struct __determinism_holder_t : __requirement
{
  static constexpr __determinism_t value = _Guarantee;

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const __get_determinism_t&) const noexcept
    -> __determinism_holder_t<_Guarantee>
  {
    return *this;
  }
};

using gpu_to_gpu_t     = __determinism_holder_t<__determinism_t::__gpu_to_gpu>;
using run_to_run_t     = __determinism_holder_t<__determinism_t::__run_to_run>;
using not_guaranteed_t = __determinism_holder_t<__determinism_t::__not_guaranteed>;

_CCCL_GLOBAL_CONSTANT gpu_to_gpu_t gpu_to_gpu{};
_CCCL_GLOBAL_CONSTANT run_to_run_t run_to_run{};
_CCCL_GLOBAL_CONSTANT not_guaranteed_t not_guaranteed{};

struct __get_determinism_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_determinism_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  [[nodiscard]]
  _CCCL_NODEBUG_API static constexpr auto query(::cuda::std::execution::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto __get_determinism = __get_determinism_t{};

} // namespace determinism

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <uscl/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_DETERMINISM_H
