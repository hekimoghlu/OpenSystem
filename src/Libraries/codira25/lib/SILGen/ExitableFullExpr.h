/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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

//===--- ExitableFullExpr.h - An exitable full-expression -------*- C++ -*-===//
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
// This file defines ExitableFullExpr, a cleanup scope RAII object
// that conveniently creates a continuation block.
//
//===----------------------------------------------------------------------===//

#ifndef EXITABLE_FULL_EXPR_H
#define EXITABLE_FULL_EXPR_H

#include "JumpDest.h"
#include "Scope.h"
#include "language/Basic/Assertions.h"

namespace language {
namespace Lowering {

/// A cleanup scope RAII object, like FullExpr, that comes with a
/// JumpDest for a continuation block.
///
/// You *must* call exit() at some point.
///
/// This scope is also exposed to the debug info.
class TOOLCHAIN_LIBRARY_VISIBILITY ExitableFullExpr {
  SILGenFunction &SGF;
  FullExpr Scope;
  JumpDest ExitDest;
public:
  explicit ExitableFullExpr(SILGenFunction &SGF, CleanupLocation loc)
    : SGF(SGF), Scope(SGF.Cleanups, loc),
      ExitDest(SGF.B.splitBlockForFallthrough(),
               SGF.Cleanups.getCleanupsDepth(), loc) {
    SGF.enterDebugScope(loc);
  }
  ~ExitableFullExpr() {
    SGF.leaveDebugScope();
  }


  JumpDest getExitDest() const { return ExitDest; }

  SILBasicBlock *exit() {
    assert(!SGF.B.hasValidInsertionPoint());
    Scope.pop();
    SGF.B.setInsertionPoint(ExitDest.getBlock());
    return ExitDest.getBlock();
  }
};

} // end namespace Lowering
} // end namespace language

#endif
