/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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
#include "B3DuplicateTails.h"

#if ENABLE(B3_JIT)

#include "B3BasicBlock.h"
#include "B3BreakCriticalEdges.h"
#include "B3FixSSA.h"
#include "B3InsertionSet.h"
#include "B3PhaseScope.h"
#include "B3Procedure.h"
#include "B3ValueInlines.h"
#include <wtf/IndexSet.h>

namespace JSC { namespace B3 {

namespace {

namespace B3DuplicateTailsInternal {
static constexpr bool verbose = false;
}

class DuplicateTails {
public:
    DuplicateTails(Procedure& proc)
        : m_proc(proc)
        , m_insertionSet(proc)
        , m_maxSize(Options::maxB3TailDupBlockSize())
        , m_maxSuccessors(Options::maxB3TailDupBlockSuccessors())
    {
    }

    void run()
    {
        // Breaking critical edges introduces blocks that jump to things. Those Jumps' successors
        // become candidates for tail duplication. Prior to critical edge breaking, some of those
        // Jumps would have been Branches, and so no tail duplication would have happened.
        breakCriticalEdges(m_proc);
        
        // Find blocks that would be candidates for tail duplication. They must be small enough
        // and they much not have too many successors.

        m_proc.resetValueOwners();

        IndexSet<BasicBlock*> candidates;

        for (BasicBlock* block : m_proc) {
            if (block->size() > m_maxSize)
                continue;
            if (block->numSuccessors() > m_maxSuccessors)
                continue;
            if (block->last()->type() != Void) // Demoting doesn't handle terminals with values.
                continue;

            bool canCopyBlock = true;
            for (Value* value : *block) {
                if (value->kind().isCloningForbidden()) {
                    canCopyBlock = false;
                    break;
                }
            }

            if (canCopyBlock)
                candidates.add(block);
        }

        // Collect the set of values that must be de-SSA'd.
        IndexSet<Value*> valuesToDemote;
        for (BasicBlock* block : m_proc) {
            for (Value* value : *block) {
                if (value->opcode() == Phi && candidates.contains(block))
                    valuesToDemote.add(value);
                for (Value* child : value->children()) {
                    if (child->owner != block && candidates.contains(child->owner))
                        valuesToDemote.add(child);
                }
            }
        }
        demoteValues(m_proc, valuesToDemote);
        if (B3DuplicateTailsInternal::verbose) {
            dataLog("Procedure after value demotion:\n");
            dataLog(m_proc);
        }

        for (BasicBlock* block : m_proc) {
            if (block->last()->opcode() != Jump)
                continue;

            BasicBlock* tail = block->successorBlock(0);
            if (!candidates.contains(tail))
                continue;

            // Don't tail duplicate a trivial self-loop, because the code below can't handle block and
            // tail being the same block.
            if (block == tail)
                continue;

            // We're about to change 'block'. Make sure that nobody duplicates block after this
            // point.
            candidates.remove(block);

            if (B3DuplicateTailsInternal::verbose)
                dataLog("Duplicating ", *tail, " into ", *block, "\n");

            block->removeLast(m_proc);

            UncheckedKeyHashMap<Value*, Value*> map;
            for (Value* value : *tail) {
                Value* clone = m_proc.clone(value);
                for (Value*& child : clone->children()) {
                    if (Value* replacement = map.get(child))
                        child = replacement;
                }
                if (value->type() != Void)
                    map.add(value, clone);
                block->append(clone);
            }
            block->successors() = tail->successors();
        }

        m_proc.resetReachability();
        m_proc.invalidateCFG();
    }
    
private:

    Procedure& m_proc;
    InsertionSet m_insertionSet;
    unsigned m_maxSize;
    unsigned m_maxSuccessors;
};

} // anonymous namespace

void duplicateTails(Procedure& proc)
{
    PhaseScope phaseScope(proc, "duplicateTails"_s);
    DuplicateTails duplicateTails(proc);
    duplicateTails.run();
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

