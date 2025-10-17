/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

//===--- MacroDiscriminatorContext.h - Macro Discriminators -----*- C++ -*-===//
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

#ifndef LANGUAGE_AST_MACRO_DISCRIMINATOR_CONTEXT_H
#define LANGUAGE_AST_MACRO_DISCRIMINATOR_CONTEXT_H

#include "language/AST/Decl.h"
#include "language/AST/Expr.h"
#include "toolchain/ADT/PointerUnion.h"

namespace language {

/// Describes the context of a macro expansion for the purpose of
/// computing macro expansion discriminators.
struct MacroDiscriminatorContext
    : public toolchain::PointerUnion<DeclContext *, FreestandingMacroExpansion *> {
  using PointerUnion::PointerUnion;

  static MacroDiscriminatorContext getParentOf(FreestandingMacroExpansion *expansion);
  static MacroDiscriminatorContext getParentOf(
      SourceLoc loc, DeclContext *origDC
  );

  /// Return the innermost declaration context that is suitable for
  /// use in identifying a macro.
  static DeclContext *getInnermostMacroContext(DeclContext *dc);
};

}

#endif // LANGUAGE_AST_MACRO_DISCRIMINATOR_CONTEXT_H
