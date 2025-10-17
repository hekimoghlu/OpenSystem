/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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

#include "AirBasicBlock.h"
#include "AirCode.h"
#include <wtf/IndexMap.h>
#include <wtf/IndexSet.h>
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 { namespace Air {

class CFG {
    WTF_MAKE_NONCOPYABLE(CFG);
    WTF_MAKE_TZONE_ALLOCATED(CFG);
public:
    typedef BasicBlock* Node;
    typedef IndexSet<BasicBlock*> Set;
    template<typename T> using Map = IndexMap<BasicBlock*, T>;
    typedef Vector<BasicBlock*, 4> List;

    CFG(Code& code)
        : m_code(code)
    {
    }

    Node root() { return m_code[0]; }

    template<typename T>
    Map<T> newMap() { return IndexMap<JSC::B3::Air::BasicBlock*, T>(m_code.size()); }

    SuccessorCollection<BasicBlock, BasicBlock::SuccessorList> successors(Node node) { return node->successorBlocks(); }
    BasicBlock::PredecessorList& predecessors(Node node) { return node->predecessors(); }

    unsigned index(Node node) const { return node->index(); }
    Node node(unsigned index) const { return m_code[index]; }
    unsigned numNodes() const { return m_code.size(); }

    PointerDump<BasicBlock> dump(Node node) const { return pointerDump(node); }

    void dump(PrintStream& out) const
    {
        m_code.dump(out);
    }

private:
    Code& m_code;
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
