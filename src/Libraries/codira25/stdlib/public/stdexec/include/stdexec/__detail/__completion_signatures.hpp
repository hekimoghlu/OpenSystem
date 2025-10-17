/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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
#pragma once

#include "__execution_fwd.hpp"

// include these after __execution_fwd.hpp
#include "__concepts.hpp"
#include "__diagnostics.hpp"
#include "__meta.hpp"

namespace stdexec {
  /////////////////////////////////////////////////////////////////////////////
  // completion_signatures
  namespace __sigs {
    template <class... _Args>
    inline constexpr bool __is_compl_sig<set_value_t(_Args...)> = true;
    template <class _Error>
    inline constexpr bool __is_compl_sig<set_error_t(_Error)> = true;
    template <>
    inline constexpr bool __is_compl_sig<set_stopped_t()> = true;

    template <class>
    inline constexpr bool __is_completion_signatures = false;
    template <class... _Sigs>
    inline constexpr bool __is_completion_signatures<completion_signatures<_Sigs...>> = true;
  } // namespace __sigs

  template <class... _Sigs>
  struct completion_signatures { };

  template <class _Completions>
  concept __valid_completion_signatures = __same_as<__ok_t<_Completions>, __msuccess>
                                       && __sigs::__is_completion_signatures<_Completions>;

  template <class _Sender, class... _Env>
  using __unrecognized_sender_error =
    __mexception<_UNRECOGNIZED_SENDER_TYPE_<>, _WITH_SENDER_<_Sender>, _WITH_ENVIRONMENT_<_Env>...>;
} // namespace stdexec
