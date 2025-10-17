/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 22, 2023.
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

//===--- SILSuccessor.cpp - Implementation of SILSuccessor.h --------------===//
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

#include "language/Basic/Assertions.h"
#include "language/SIL/SILSuccessor.h"
#include "language/SIL/SILBasicBlock.h"
using namespace language;

void SILSuccessor::operator=(SILBasicBlock *BB) {
  // If we're not changing anything, we're done.
  if (SuccessorBlock == BB) return;
  
  assert(ContainingInst &&"init method not called after default construction?");
  
  // If we were already pointing to a basic block, remove ourself from its
  // predecessor list.
  if (SuccessorBlock) {
    *Prev = Next;
    if (Next) Next->Prev = Prev;
  }
  
  // If we have a successor, add ourself to its prev list.
  if (BB) {
    Prev = &BB->PredList;
    Next = BB->PredList;
    if (Next) Next->Prev = &Next;
    BB->PredList = this;
  }
  
  SuccessorBlock = BB;
}
