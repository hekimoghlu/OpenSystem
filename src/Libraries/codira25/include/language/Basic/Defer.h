/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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

//===--- Defer.h - 'defer' helper macro -------------------------*- C++ -*-===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//
//
// This file defines a 'LANGUAGE_DEFER' macro for performing a cleanup on any exit
// out of a scope.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_BASIC_DEFER_H
#define LANGUAGE_BASIC_DEFER_H

#include "toolchain/ADT/ScopeExit.h"

namespace language {
  namespace detail {
    struct DeferTask {};
    template<typename F>
    auto operator+(DeferTask, F &&fn) ->
        decltype(toolchain::make_scope_exit(std::forward<F>(fn))) {
      return toolchain::make_scope_exit(std::forward<F>(fn));
    }
  }
} // end namespace language


#define DEFER_CONCAT_IMPL(x, y) x##y
#define DEFER_MACRO_CONCAT(x, y) DEFER_CONCAT_IMPL(x, y)

/// This macro is used to register a function / lambda to be run on exit from a
/// scope.  Its typical use looks like:
///
///   LANGUAGE_DEFER {
///     stuff
///   };
///
#define LANGUAGE_DEFER                                                            \
  auto DEFER_MACRO_CONCAT(defer_func, __COUNTER__) =                           \
      ::language::detail::DeferTask() + [&]()

#endif // LANGUAGE_BASIC_DEFER_H
