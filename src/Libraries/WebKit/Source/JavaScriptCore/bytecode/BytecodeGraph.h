/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 11, 2025.
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

#include "BytecodeBasicBlock.h"
#include "BytecodeDumper.h"
#include <wtf/IndexedContainerIterator.h>
#include <wtf/IteratorRange.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC {

class BytecodeGraph {
    WTF_MAKE_NONCOPYABLE(BytecodeGraph);
    WTF_MAKE_TZONE_ALLOCATED(BytecodeGraph);
public:
    using BasicBlockType = JSBytecodeBasicBlock;
    using BasicBlocksVector = typename BasicBlockType::BasicBlockVector;
    using InstructionStreamType = typename BasicBlockType::InstructionStreamType;

    typedef WTF::IndexedContainerIterator<BytecodeGraph> iterator;

    template <typename CodeBlockType>
    inline BytecodeGraph(CodeBlockType*, const InstructionStreamType&);

    WTF::IteratorRange<BasicBlocksVector::reverse_iterator> basicBlocksInReverseOrder()
    {
        return WTF::makeIteratorRange(m_basicBlocks.rbegin(), m_basicBlocks.rend());
    }

    static bool blockContainsBytecodeOffset(const BasicBlockType& block, typename InstructionStreamType::Offset bytecodeOffset)
    {
        unsigned leaderOffset = block.leaderOffset();
        return bytecodeOffset >= leaderOffset && bytecodeOffset < leaderOffset + block.totalLength();
    }

    BasicBlockType* findBasicBlockForBytecodeOffset(typename InstructionStreamType::Offset bytecodeOffset)
    {
        /*
            for (unsigned i = 0; i < m_basicBlocks.size(); i++) {
                if (blockContainsBytecodeOffset(m_basicBlocks[i], bytecodeOffset))
                    return &m_basicBlocks[i];
            }
            return 0;
        */

        BasicBlockType* basicBlock = approximateBinarySearch<BasicBlockType, unsigned>(m_basicBlocks, m_basicBlocks.size(), bytecodeOffset, [] (BasicBlockType* basicBlock) { return basicBlock->leaderOffset(); });
        // We found the block we were looking for.
        if (blockContainsBytecodeOffset(*basicBlock, bytecodeOffset))
            return basicBlock;

        // Basic block is to the left of the returned block.
        if (bytecodeOffset < basicBlock->leaderOffset()) {
            ASSERT(basicBlock - 1 >= m_basicBlocks.data());
            ASSERT(blockContainsBytecodeOffset(basicBlock[-1], bytecodeOffset));
            return &basicBlock[-1];
        }

        // Basic block is to the right of the returned block.
        ASSERT(&basicBlock[1] <= &m_basicBlocks.last());
        ASSERT(blockContainsBytecodeOffset(basicBlock[1], bytecodeOffset));
        return &basicBlock[1];
    }

    BasicBlockType* findBasicBlockWithLeaderOffset(typename InstructionStreamType::Offset leaderOffset)
    {
        return tryBinarySearch<BasicBlockType, unsigned>(m_basicBlocks, m_basicBlocks.size(), leaderOffset, [] (BasicBlockType* basicBlock) { return basicBlock->leaderOffset(); });
    }

    unsigned size() const { return m_basicBlocks.size(); }
    BasicBlockType& at(unsigned index) const { return const_cast<BytecodeGraph*>(this)->m_basicBlocks[index]; }
    BasicBlockType& operator[](unsigned index) const { return at(index); }

    iterator begin() { return iterator(*this, 0); }
    iterator end() { return iterator(*this, size()); }
    BasicBlockType& first() { return at(0); }
    BasicBlockType& last() { return at(size() - 1); }


    template <typename CodeBlockType>
    void dump(CodeBlockType* codeBlock, const InstructionStreamType& instructions, std::optional<Vector<Operands<SpeculatedType>>> speculationAtHead, PrintStream& printer = WTF::dataFile())
    {
        CodeBlockBytecodeDumper<CodeBlockType>::dumpGraph(codeBlock, instructions, *this, speculationAtHead, printer);
    }

private:
    BasicBlocksVector m_basicBlocks;
};


template<typename CodeBlockType>
BytecodeGraph::BytecodeGraph(CodeBlockType* codeBlock, const InstructionStreamType& instructions)
    : m_basicBlocks(BasicBlockType::compute(codeBlock, instructions))
{
    ASSERT(m_basicBlocks.size());
}

} // namespace JSC

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
