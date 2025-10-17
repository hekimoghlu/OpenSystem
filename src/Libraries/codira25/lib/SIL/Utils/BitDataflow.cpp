/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 6, 2024.
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

//===--- BitDataflow.cpp --------------------------------------------------===//
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

#define DEBUG_TYPE "bit-dataflow"
#include "language/Basic/Assertions.h"
#include "language/SIL/BitDataflow.h"
#include "language/Basic/SmallBitVector.h"
#include "language/SIL/MemoryLocations.h"
#include "language/SIL/SILBasicBlock.h"
#include "language/SIL/SILFunction.h"
#include "toolchain/Support/raw_ostream.h"

using namespace language;

BitDataflow::BitDataflow(SILFunction *function, unsigned numLocations) :
  blockStates(function, [numLocations](SILBasicBlock *block) {
    return BlockState(numLocations);
  }) {}

void BitDataflow::entryReachabilityAnalysis() {
  toolchain::SmallVector<SILBasicBlock *, 16> workList;
  auto entry = blockStates.entry();
  entry.data.reachableFromEntry = true;
  workList.push_back(&entry.block);

  while (!workList.empty()) {
    SILBasicBlock *block = workList.pop_back_val();
    for (SILBasicBlock *succ : block->getSuccessorBlocks()) {
      BlockState &succState = blockStates[succ];
      if (!succState.reachableFromEntry) {
        succState.reachableFromEntry = true;
        workList.push_back(succ);
      }
    }
  }
}

void BitDataflow::exitReachableAnalysis() {
  toolchain::SmallVector<SILBasicBlock *, 16> workList;
  for (auto bd : blockStates) {
    if (bd.block.getTerminator()->isFunctionExiting()) {
      bd.data.exitReachability = ExitReachability::ReachesExit;
      workList.push_back(&bd.block);
    } else if (isa<UnreachableInst>(bd.block.getTerminator())) {
      bd.data.exitReachability = ExitReachability::ReachesUnreachable;
      workList.push_back(&bd.block);
    }
  }
  while (!workList.empty()) {
    SILBasicBlock *block = workList.pop_back_val();
    BlockState &state = blockStates[block];
    for (SILBasicBlock *pred : block->getPredecessorBlocks()) {
      BlockState &predState = blockStates[pred];
      if (predState.exitReachability < state.exitReachability) {
        // As there are 3 states, each block can be put into the workList 2
        // times maximum.
        predState.exitReachability = state.exitReachability;
        workList.push_back(pred);
      }
    }
  }
}

void BitDataflow::solveForward(JoinOperation join) {
  // Pretty standard data flow solving.
  bool changed = false;
  bool firstRound = true;
  do {
    changed = false;
    for (auto bd : blockStates) {
      Bits bits = bd.data.entrySet;
      assert(!bits.empty());
      for (SILBasicBlock *pred : bd.block.getPredecessorBlocks()) {
        join(bits, blockStates[pred].exitSet);
      }
      if (firstRound || bits != bd.data.entrySet) {
        changed = true;
        bd.data.entrySet = bits;
        bits |= bd.data.genSet;
        bits.reset(bd.data.killSet);
        bd.data.exitSet = bits;
      }
    }
    firstRound = false;
  } while (changed);
}

void BitDataflow::solveForwardWithIntersect() {
  solveForward([](Bits &entry, const Bits &predExit){
    entry &= predExit;
  });
}

void BitDataflow::solveForwardWithUnion() {
  solveForward([](Bits &entry, const Bits &predExit){
    entry |= predExit;
  });
}

void BitDataflow::solveBackward(JoinOperation join) {
  // Pretty standard data flow solving.
  bool changed = false;
  bool firstRound = true;
  do {
    changed = false;
    for (auto bd : toolchain::reverse(blockStates)) {
      Bits bits = bd.data.exitSet;
      assert(!bits.empty());
      for (SILBasicBlock *succ : bd.block.getSuccessorBlocks()) {
        join(bits, blockStates[succ].entrySet);
      }
      if (firstRound || bits != bd.data.exitSet) {
        changed = true;
        bd.data.exitSet = bits;
        bits |= bd.data.genSet;
        bits.reset(bd.data.killSet);
        bd.data.entrySet = bits;
      }
    }
    firstRound = false;
  } while (changed);
}

void BitDataflow::solveBackwardWithIntersect() {
  solveBackward([](Bits &entry, const Bits &predExit){
    entry &= predExit;
  });
}

void BitDataflow::solveBackwardWithUnion() {
  solveBackward([](Bits &entry, const Bits &predExit){
    entry |= predExit;
  });
}

void BitDataflow::dump() const {
    for (auto bd : blockStates) {
    toolchain::dbgs() << "bb" << bd.block.getDebugID() << ":\n"
                 << "    entry: " << bd.data.entrySet << '\n'
                 << "    gen:   " << bd.data.genSet << '\n'
                 << "    kill:  " << bd.data.killSet << '\n'
                 << "    exit:  " << bd.data.exitSet << '\n';
  }
}
