/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 11, 2021.
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
#include "B3EnsureLoopPreHeaders.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlockInlines.h"
#include "B3BlockInsertionSet.h"
#include "B3NaturalLoops.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

bool ensureLoopPreHeaders(Procedure& proc)
{
    NaturalLoops& loops = proc.naturalLoops();
    
    BlockInsertionSet insertionSet(proc);
    
    for (unsigned loopIndex = loops.numLoops(); loopIndex--;) {
        const NaturalLoop& loop = loops.loop(loopIndex);
        
        Vector<BasicBlock*, 4> outOfBodyPredecessors;
        double totalFrequency = 0;
        for (BasicBlock* predecessor : loop.header()->predecessors()) {
            if (loops.belongsTo(predecessor, loop))
                continue;
            
            outOfBodyPredecessors.append(predecessor);
            totalFrequency += predecessor->frequency();
        }
        
        if (outOfBodyPredecessors.size() <= 1)
            continue;
        
        BasicBlock* preHeader = insertionSet.insertBefore(loop.header(), totalFrequency);
        preHeader->appendNew<Value>(proc, Jump, loop.header()->at(0)->origin());
        preHeader->setSuccessors(FrequentedBlock(loop.header()));
        
        for (BasicBlock* predecessor : outOfBodyPredecessors) {
            predecessor->replaceSuccessor(loop.header(), preHeader);
            preHeader->addPredecessor(predecessor);
            loop.header()->removePredecessor(predecessor);
        }
        
        loop.header()->addPredecessor(preHeader);
    }
    
    if (insertionSet.execute()) {
        proc.invalidateCFG();
        return true;
    }
    
    return false;
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

