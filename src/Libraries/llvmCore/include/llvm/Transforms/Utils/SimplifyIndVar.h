/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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

//===-- llvm/Transforms/Utils/SimplifyIndVar.h - Indvar Utils ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines in interface for induction variable simplification. It does
// not define any actual pass or policy, but provides a single function to
// simplify a loop's induction variables based on ScalarEvolution.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_SIMPLIFYINDVAR_H
#define LLVM_TRANSFORMS_UTILS_SIMPLIFYINDVAR_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ValueHandle.h"

namespace llvm {

class CastInst;
class IVUsers;
class Loop;
class LPPassManager;
class PHINode;
class ScalarEvolution;

/// Interface for visiting interesting IV users that are recognized but not
/// simplified by this utility.
class IVVisitor {
  virtual void anchor();
public:
  virtual ~IVVisitor() {}
  virtual void visitCast(CastInst *Cast) = 0;
};

/// simplifyUsersOfIV - Simplify instructions that use this induction variable
/// by using ScalarEvolution to analyze the IV's recurrence.
bool simplifyUsersOfIV(PHINode *CurrIV, ScalarEvolution *SE, LPPassManager *LPM,
                       SmallVectorImpl<WeakVH> &Dead, IVVisitor *V = NULL);

/// SimplifyLoopIVs - Simplify users of induction variables within this
/// loop. This does not actually change or add IVs.
bool simplifyLoopIVs(Loop *L, ScalarEvolution *SE, LPPassManager *LPM,
                     SmallVectorImpl<WeakVH> &Dead);

} // namespace llvm

#endif
