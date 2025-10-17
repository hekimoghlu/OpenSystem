/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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

#include "../stdexec/__detail/__scope.hpp"

namespace exec {

  template <class _Fn, class... _Ts>
    requires stdexec::__nothrow_callable<_Fn, _Ts...>
  struct scope_guard {
    stdexec::__scope_guard<_Fn, _Ts...> __guard_;

    void dismiss() noexcept {
      __guard_.__dismiss();
    }
  };
  template <class _Fn, class... _Ts>
  scope_guard(_Fn, _Ts...) -> scope_guard<_Fn, _Ts...>;

} // namespace exec
