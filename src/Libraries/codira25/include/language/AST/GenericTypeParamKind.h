/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 23, 2022.
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

//===--- GenericTypeParamKind.h ---------------------------------*- C++ -*-===//
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
///
/// This file defines the `GenericTypeParamKind` enum.
///
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_GENERICTYPEPARAMKIND_H
#define LANGUAGE_AST_GENERICTYPEPARAMKIND_H

/// This header is included in a bridging header. Be *very* careful with what
/// you include here! See include caveats in `ASTBridging.h`.
#include "language/Basic/LanguageBridging.h"
#include <stdint.h>

namespace language {
/// Describes the kind of a generic type parameter that occurs within generic
/// parameter lists.
enum class ENUM_EXTENSIBILITY_ATTR(closed) GenericTypeParamKind : uint8_t {
  /// A regular generic type parameter: 'T'
  Type LANGUAGE_NAME("type"),
  /// A generic parameter pack: 'each T'
  Pack LANGUAGE_NAME("pack"),
  /// A generic value parameter: 'let T'
  Value LANGUAGE_NAME("value")
};

} // namespace language
#endif // #ifndef LANGUAGE_AST_GENERICTYPEPARAMKIND_H
