/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include <wtf/Insertion.h>
#include <wtf/Vector.h>

namespace JSC { namespace DFG {

typedef WTF::Insertion<std::unique_ptr<BasicBlock>> BlockInsertion;

class BlockInsertionSet {
public:
    BlockInsertionSet(Graph&);
    ~BlockInsertionSet();
    
    void insert(BlockInsertion&&);
    void insert(size_t index, std::unique_ptr<BasicBlock>&&);
    BasicBlock* insert(size_t index, float executionCount);
    BasicBlock* insertBefore(BasicBlock* before, float executionCount);

    bool execute();

private:
    Graph& m_graph;
    Vector<BlockInsertion, 8> m_insertions;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
