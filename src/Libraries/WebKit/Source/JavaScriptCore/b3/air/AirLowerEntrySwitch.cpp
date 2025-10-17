/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 6, 2023.
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
#include "AirLowerEntrySwitch.h"

#if ENABLE(B3_JIT)

#include "AirBlockWorklist.h"
#include "AirCode.h"
#include "AirPhaseScope.h"
#include "B3Procedure.h"

namespace JSC { namespace B3 { namespace Air {

void lowerEntrySwitch(Code& code)
{
    PhaseScope phaseScope(code, "lowerEntrySwitch"_s);
    
    // Figure out the set of blocks that should be duplicated.
    BlockWorklist worklist;
    for (BasicBlock* block : code) {
        if (block->last().kind.opcode == EntrySwitch)
            worklist.push(block);
    }
    
    // It's possible that we don't have any EntrySwitches. That's fine.
    if (worklist.seen().isEmpty()) {
        Vector<FrequentedBlock> entrypoints(code.proc().numEntrypoints(), FrequentedBlock(code[0]));
        code.setEntrypoints(WTFMove(entrypoints));
        return;
    }
    
    while (BasicBlock* block = worklist.pop())
        worklist.pushAll(block->predecessors());
    
    RELEASE_ASSERT(worklist.saw(code[0]));
    
    Vector<FrequencyClass> entrypointFrequencies(code.proc().numEntrypoints(), FrequencyClass::Rare);
    for (BasicBlock* block : code) {
        if (block->last().kind.opcode != EntrySwitch)
            continue;
        for (unsigned entrypointIndex = code.proc().numEntrypoints(); entrypointIndex--;) {
            entrypointFrequencies[entrypointIndex] = maxFrequency(
                entrypointFrequencies[entrypointIndex],
                block->successor(entrypointIndex).frequency());
        }
    }
    
    auto fixEntrySwitch = [&] (BasicBlock* block, unsigned entrypointIndex) {
        if (block->last().kind.opcode != EntrySwitch)
            return;
        FrequentedBlock target = block->successor(entrypointIndex);
        block->last().kind.opcode = Jump;
        block->successors().resize(1);
        block->successor(0) = target;
    };
    
    // Now duplicate them.
    Vector<FrequentedBlock> entrypoints;
    entrypoints.append(FrequentedBlock(code[0], entrypointFrequencies[0]));
    IndexMap<BasicBlock*, BasicBlock*> map(code.size());
    for (unsigned entrypointIndex = 1; entrypointIndex < code.proc().numEntrypoints(); ++entrypointIndex) {
        map.clear();
        for (BasicBlock* block : worklist.seen().values(code))
            map[block] = code.addBlock(block->frequency());
        entrypoints.append(FrequentedBlock(map[code[0]], entrypointFrequencies[entrypointIndex]));
        for (BasicBlock* block : worklist.seen().values(code)) {
            BasicBlock* newBlock = map[block];
            for (const Inst& inst : *block)
                newBlock->appendInst(inst);
            newBlock->successors() = block->successors();
            for (BasicBlock*& successor : newBlock->successorBlocks()) {
                if (BasicBlock* replacement = map[successor])
                    successor = replacement;
            }
            fixEntrySwitch(newBlock, entrypointIndex);
        }
    }
    for (BasicBlock* block : worklist.seen().values(code))
        fixEntrySwitch(block, 0);
    
    code.setEntrypoints(WTFMove(entrypoints));
    code.resetReachability();
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)


