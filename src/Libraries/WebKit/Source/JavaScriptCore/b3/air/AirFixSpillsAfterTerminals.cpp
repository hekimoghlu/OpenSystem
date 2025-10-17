/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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
#include "config.h"
#include "AirFixSpillsAfterTerminals.h"

#if ENABLE(B3_JIT)

#include "AirBlockInsertionSet.h"
#include "AirCode.h"
#include "AirInsertionSet.h"

namespace JSC { namespace B3 { namespace Air {

void fixSpillsAfterTerminals(Code& code)
{
    // Because there may be terminals that produce values, IRC may
    // want to spill those terminals. It'll happen to spill it after
    // the terminal. If we left the graph in this state, it'd be invalid
    // because a terminal must be the last instruction in a block.
    // We fix that here.

    BlockInsertionSet blockInsertionSet(code);
    InsertionSet insertionSet(code);

    for (BasicBlock* block : code) {
        unsigned terminalIndex = block->size();
        bool foundTerminal = false;
        while (terminalIndex--) {
            if (block->at(terminalIndex).isTerminal()) {
                foundTerminal = true;
                break;
            }
        }
        ASSERT_UNUSED(foundTerminal, foundTerminal);

        if (terminalIndex == block->size() - 1)
            continue;

        // There must be instructions after the terminal because it's not the last instruction.
        ASSERT(terminalIndex < block->size() - 1);
        Vector<Inst, 1> instsToMove;
        for (unsigned i = terminalIndex + 1; i < block->size(); i++)
            instsToMove.append(block->at(i));
        RELEASE_ASSERT(instsToMove.size());

        for (FrequentedBlock& frequentedSuccessor : block->successors()) {
            BasicBlock* successor = frequentedSuccessor.block();
            // If successor's only predecessor is block, we can plant the spill inside
            // the successor. Otherwise, we must split the critical edge and create
            // a new block for the spill.
            if (successor->numPredecessors() == 1) {
                insertionSet.insertInsts(0, instsToMove);
                insertionSet.execute(successor);
            } else {
                BasicBlock* newBlock = blockInsertionSet.insertBefore(successor, successor->frequency());
                for (const Inst& inst : instsToMove)
                    newBlock->appendInst(inst);
                newBlock->appendInst(Inst(Jump, instsToMove.last().origin));
                newBlock->successors().append(successor);
                frequentedSuccessor.block() = newBlock;
            }
        }

        block->resize(terminalIndex + 1);
    }

    if (blockInsertionSet.execute())
        code.resetReachability();
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

