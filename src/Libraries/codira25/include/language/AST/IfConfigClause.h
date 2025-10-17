/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

//===--- IfConfigClause.h ---------------------------------------*- C++ -*-===//
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
// This file defines the IfConfigClause.
//
//===----------------------------------------------------------------------===//

#ifndef LANGUAGE_AST_IFCONFIGCLAUSE_H
#define LANGUAGE_AST_IFCONFIGCLAUSE_H

#include "toolchain/ADT/ArrayRef.h"

namespace language {
  class Expr;
  class SourceLoc;
  struct ASTNode;

/// This represents one part of a #if block.  If the condition field is
/// non-null, then this represents a #if or a #elseif, otherwise it represents
/// an #else block.
struct IfConfigClause {
  /// The location of the #if, #elseif, or #else keyword.
  SourceLoc Loc;
  
  /// The condition guarding this #if or #elseif block.  If this is null, this
  /// is a #else clause.
  Expr *Cond;
  
  /// Elements inside the clause
  ArrayRef<ASTNode> Elements;

  /// True if this is the active clause of the #if block.
  const bool isActive;

  IfConfigClause(SourceLoc Loc, Expr *Cond,
                 ArrayRef<ASTNode> Elements, bool isActive)
    : Loc(Loc), Cond(Cond), Elements(Elements), isActive(isActive) {
  }
};

} // end namespace language

#endif
