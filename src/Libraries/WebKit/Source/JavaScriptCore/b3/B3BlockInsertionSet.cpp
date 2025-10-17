/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#include "B3BlockInsertionSet.h"

#if ENABLE(B3_JIT)

#include "B3InsertionSet.h"
#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

BlockInsertionSet::BlockInsertionSet(Procedure &proc)
    : GenericBlockInsertionSet(proc.m_blocks)
    , m_proc(proc)
{
}

BlockInsertionSet::~BlockInsertionSet() = default;

BasicBlock* BlockInsertionSet::splitForward(
    BasicBlock* block, unsigned& valueIndex, InsertionSet* insertionSet, double frequency)
{
    Value* value = block->at(valueIndex);

    // Create a new block that will go just before 'block', and make it contain everything prior
    // to 'valueIndex'.
    BasicBlock* result = insertBefore(block, frequency);
    result->m_values.resize(valueIndex + 1);
    for (unsigned i = valueIndex; i--;)
        result->m_values[i] = block->m_values[i];

    // Make the new block jump to 'block'.
    result->m_values[valueIndex] = m_proc.add<Value>(Jump, value->origin());
    result->setSuccessors(FrequentedBlock(block));

    // If we had inserted things into 'block' before this, execute those insertions now.
    if (insertionSet)
        insertionSet->execute(result);

    // Remove everything prior to 'valueIndex' from 'block', since those things are now in the
    // new block.
    block->m_values.remove(0, valueIndex);

    // This is being used in a forward loop over 'block'. Update the index of the loop so that
    // it can continue to the next block.
    valueIndex = 0;

    // Fixup the predecessors of 'block'. They now must jump to the new block.
    result->predecessors() = WTFMove(block->predecessors());
    block->addPredecessor(result);
    for (BasicBlock* predecessor : result->predecessors())
        predecessor->replaceSuccessor(block, result);

    return result;
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

