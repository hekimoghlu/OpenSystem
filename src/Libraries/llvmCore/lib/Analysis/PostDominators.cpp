/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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

//===- PostDominators.cpp - Post-Dominator Calculation --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the post-dominator construction algorithms.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "postdomtree"

#include "llvm/Analysis/PostDominators.h"
#include "llvm/Instructions.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/Assembly/Writer.h"
#include "llvm/Analysis/DominatorInternals.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
//  PostDominatorTree Implementation
//===----------------------------------------------------------------------===//

char PostDominatorTree::ID = 0;
INITIALIZE_PASS(PostDominatorTree, "postdomtree",
                "Post-Dominator Tree Construction", true, true)

bool PostDominatorTree::runOnFunction(Function &F) {
  DT->recalculate(F);
  return false;
}

PostDominatorTree::~PostDominatorTree() {
  delete DT;
}

void PostDominatorTree::print(raw_ostream &OS, const Module *) const {
  DT->print(OS);
}


FunctionPass* llvm::createPostDomTree() {
  return new PostDominatorTree();
}

