/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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

//===--- SILDebugScope.cpp - DebugScopes for SIL code ---------------------===//
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
/// \file
///
/// This file defines a container for scope information used to
/// generate debug info.
///
//===----------------------------------------------------------------------===//

#include "language/Basic/Assertions.h"
#include "language/SIL/SILDebugScope.h"
#include "language/SIL/SILFunction.h"

using namespace language;

SILDebugScope::SILDebugScope(SILLocation Loc, SILFunction *SILFn,
                             const SILDebugScope *ParentScope,
                             const SILDebugScope *InlinedCallSite)
    : Loc(Loc), InlinedCallSite(InlinedCallSite) {
  if (ParentScope)
    Parent = ParentScope;
  else {
    assert(SILFn && "no parent provided");
    Parent = SILFn;
  }
}

SILDebugScope::SILDebugScope(SILLocation Loc)
    : Loc(Loc), InlinedCallSite(nullptr) {}

SILFunction *SILDebugScope::getInlinedFunction() const {
  if (Parent.isNull())
    return nullptr;

  const SILDebugScope *Scope = this;
  while (Scope->Parent.is<const SILDebugScope *>())
    Scope = Scope->Parent.get<const SILDebugScope *>();
  assert(Scope->Parent.is<SILFunction *>() && "orphaned scope");
  return Scope->Parent.get<SILFunction *>();
}

SILFunction *SILDebugScope::getParentFunction() const {
  if (InlinedCallSite)
    return InlinedCallSite->getParentFunction();
  if (auto *ParentScope = Parent.dyn_cast<const SILDebugScope *>())
    return ParentScope->getParentFunction();
  return Parent.get<SILFunction *>();
}

/// Determine whether an instruction may not have a SILDebugScope.
bool language::maybeScopeless(const SILInstruction &inst) {
  if (inst.getFunction()->isBare())
    return true;
  return !isa<DebugValueInst>(inst);
}
