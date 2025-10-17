/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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

//===--- LifetimeAnnotation.h - Lifetime-affecting attributes ---*- C++ -*-===//
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
// Defines a simple type-safe wrapper around the annotations that affect value
// lifetimes.  Used for both Decls and SILFunctionArguments.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_LIFETIMEANNOTATION_H
#define LANGUAGE_AST_LIFETIMEANNOTATION_H

#include <cstdint>

namespace language {

/// The annotation on a value (or type) explicitly indicating the lifetime that
/// it (or its instances) should have.
///
/// A LifetimeAnnotation is one of the following three values:
///
/// 1) None: No annotation has been applied.
/// 2) EagerMove: The @_eagerMove attribute has been applied.
/// 3) NoEagerMove: The @_noEagerMove attribute has been applied.
struct LifetimeAnnotation {
  enum Case : uint8_t {
    None,
    EagerMove,
    Lexical,
  } value;

  LifetimeAnnotation(Case newValue) : value(newValue) {}

  operator Case() const { return value; }

  bool isSome() { return value != None; }
};

} // namespace language

#endif
