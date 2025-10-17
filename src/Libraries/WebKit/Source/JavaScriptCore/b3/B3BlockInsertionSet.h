/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
#pragma once

#if ENABLE(B3_JIT)

#include "B3GenericBlockInsertionSet.h"
#include "B3Procedure.h"
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class InsertionSet;

typedef GenericBlockInsertionSet<BasicBlock>::BlockInsertion BlockInsertion;

class BlockInsertionSet : public GenericBlockInsertionSet<BasicBlock> {
public:
    BlockInsertionSet(Procedure&);
    ~BlockInsertionSet();
    
    // A helper to split a block when forward iterating over it. It creates a new block to hold
    // everything before the instruction at valueIndex. The current block is left with
    // everything at and after valueIndex. If the optional InsertionSet is provided, it will get
    // executed on the newly created block - this makes sense if you had previously inserted
    // things into the original block, since the newly created block will be indexed identically
    // to how this block was indexed for all values prior to valueIndex. After this runs, it sets
    // valueIndex to zero. This allows you to use this method for things like:
    //
    // for (unsigned valueIndex = 0; valueIndex < block->size(); ++valueIndex) {
    //     Value* value = block->at(valueIndex);
    //     if (value->opcode() == Foo) {
    //         BasicBlock* predecessor =
    //             m_blockInsertionSet.splitForward(block, valueIndex, &m_insertionSet);
    //         ... // Now you can append to predecessor, insert new blocks before 'block', and
    //         ... // you can use m_insertionSet to insert more thing before 'value'.
    //         predecessor->updatePredecessorsAfter();
    //     }
    // }
    //
    // Note how usually this idiom ends in a all to updatePredecessorsAftter(), which ensures
    // that the predecessors involved in any of the new control flow that you've created are up
    // to date.
    BasicBlock* splitForward(
        BasicBlock*, unsigned& valueIndex, InsertionSet* = nullptr,
        double frequency = PNaN);

private:
    Procedure& m_proc;
};

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
