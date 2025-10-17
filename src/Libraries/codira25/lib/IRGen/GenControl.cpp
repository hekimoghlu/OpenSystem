/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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

//===--- GenControl.cpp - IR Generation for Control Flow ------------------===//
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
//  This file implements general IR generation for control flow.
//
//===----------------------------------------------------------------------===//

#include "toolchain/ADT/STLExtras.h"
#include "toolchain/IR/Function.h"
#include "IRGenFunction.h"
#include "IRGenModule.h"
#include "language/Basic/Assertions.h"

using namespace language;
using namespace irgen;

/// Insert the given basic block after the IP block and move the
/// insertion point to it.  Only valid if the IP is valid.
void IRBuilder::emitBlock(toolchain::BasicBlock *BB) {
  assert(ClearedIP == nullptr);
  toolchain::BasicBlock *CurBB = GetInsertBlock();
  assert(CurBB && "current insertion point is invalid");
  CurBB->getParent()->insert(std::next(CurBB->getIterator()), BB);
  IRBuilderBase::SetInsertPoint(BB);
}

/// Create a new basic block with the given name.  The block is not
/// automatically inserted into the function.
toolchain::BasicBlock *
IRGenFunction::createBasicBlock(const toolchain::Twine &Name) {
  return toolchain::BasicBlock::Create(IGM.getLLVMContext(), Name);
}

