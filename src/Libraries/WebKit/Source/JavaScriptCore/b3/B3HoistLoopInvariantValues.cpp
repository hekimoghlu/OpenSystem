/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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
#include "B3HoistLoopInvariantValues.h"

#if ENABLE(B3_JIT)

#include "B3BackwardsDominators.h"
#include "B3Dominators.h"
#include "B3EnsureLoopPreHeaders.h"
#include "B3NaturalLoops.h"
#include "B3PhaseScope.h"
#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"
#include <wtf/RangeSet.h>

namespace JSC { namespace B3 {

bool hoistLoopInvariantValues(Procedure& proc)
{
    PhaseScope phaseScope(proc, "hoistLoopInvariantValues"_s);
    
    ensureLoopPreHeaders(proc);
    
    NaturalLoops& loops = proc.naturalLoops();
    if (!loops.numLoops())
        return false;

    proc.resetValueOwners();
    Dominators& dominators = proc.dominators();
    BackwardsDominators& backwardsDominators = proc.backwardsDominators();
    
    // FIXME: We should have a reusable B3::EffectsSet data structure.
    // https://bugs.webkit.org/show_bug.cgi?id=174762
    struct LoopData {
        RangeSet<HeapRange> writes;
        bool writesLocalState { false };
        bool writesPinned { false };
        bool sideExits { false };
        BasicBlock* preHeader { nullptr };
    };
    
    IndexMap<NaturalLoop, LoopData> data(loops.numLoops());
    
    for (unsigned loopIndex = loops.numLoops(); loopIndex--;) {
        const NaturalLoop& loop = loops.loop(loopIndex);
        for (BasicBlock* predecessor : loop.header()->predecessors()) {
            if (loops.belongsTo(predecessor, loop))
                continue;
            RELEASE_ASSERT(!data[loop].preHeader);
            data[loop].preHeader = predecessor;
        }
    }
    
    for (BasicBlock* block : proc) {
        const NaturalLoop* loop = loops.innerMostLoopOf(block);
        if (!loop)
            continue;
        for (Value* value : *block) {
            Effects effects = value->effects();
            data[*loop].writes.add(effects.writes);
            data[*loop].writesLocalState |= effects.writesLocalState;
            data[*loop].writesPinned |= effects.writesPinned;
            data[*loop].sideExits |= effects.exitsSideways;
        }
    }
    
    for (unsigned loopIndex = loops.numLoops(); loopIndex--;) {
        const NaturalLoop& loop = loops.loop(loopIndex);
        for (const NaturalLoop* current = loops.innerMostOuterLoop(loop); current; current = loops.innerMostOuterLoop(*current)) {
            data[*current].writes.addAll(data[loop].writes);
            data[*current].writesLocalState |= data[loop].writesLocalState;
            data[*current].writesPinned |= data[loop].writesPinned;
            data[*current].sideExits |= data[loop].sideExits;
        }
    }
    
    bool changed = false;
    
    // Pre-order ensures that we visit our dominators before we visit ourselves. Otherwise we'd miss some
    // hoisting opportunities in complex CFGs.
    for (BasicBlock* block : proc.blocksInPreOrder()) {
        Vector<const NaturalLoop*> blockLoops = loops.loopsOf(block);
        if (blockLoops.isEmpty())
            continue;
        
        for (Value*& value : *block) {
            Effects effects = value->effects();
            
            // We never hoist write effects or control constructs.
            if (effects.mustExecute())
                continue;

            // Try outermost loop first.
            for (unsigned i = blockLoops.size(); i--;) {
                const NaturalLoop& loop = *blockLoops[i];
                
                bool ok = true;
                for (Value* child : value->children()) {
                    if (!dominators.dominates(child->owner, data[loop].preHeader)) {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                    continue;
                
                if (effects.controlDependent) {
                    if (!backwardsDominators.dominates(block, data[loop].preHeader))
                        continue;
                    
                    // FIXME: This is super conservative. In reality, we just need to make sure that there
                    // aren't any side exits between here and the pre-header according to backwards search.
                    // https://bugs.webkit.org/show_bug.cgi?id=174763
                    if (data[loop].sideExits)
                        continue;
                }
                
                if (effects.readsPinned && data[loop].writesPinned)
                    continue;
                
                if (effects.readsLocalState && data[loop].writesLocalState)
                    continue;
                
                if (data[loop].writes.overlaps(effects.reads))
                    continue;
                
                data[loop].preHeader->appendNonTerminal(value);
                value = proc.add<Value>(Nop, Void, value->origin());
                changed = true;
            }
        }
    }
    
    return changed;
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

