/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

#include "InstructionStream.h"
#include "Opcode.h"
#include <limits.h>
#include <wtf/FastBitVector.h>
#include <wtf/Vector.h>

namespace JSC {

class BytecodeGraph;
class CodeBlock;
class UnlinkedCodeBlock;
class UnlinkedCodeBlockGenerator;
template<typename> struct Instruction;

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(BytecodeBasicBlock);

template<typename OpcodeTraits>
class BytecodeBasicBlock {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(BytecodeBasicBlock);
    WTF_MAKE_NONCOPYABLE(BytecodeBasicBlock);
    friend class BytecodeGraph;
public:
    using BasicBlockVector = Vector<BytecodeBasicBlock, 0, UnsafeVectorOverflow, 16, BytecodeBasicBlockMalloc>;
    using InstructionType = BaseInstruction<OpcodeTraits>;
    using InstructionStreamType = InstructionStream<InstructionType>;
    static_assert(maxBytecodeStructLength <= UINT8_MAX);
    enum SpecialBlockType { EntryBlock, ExitBlock };
    inline BytecodeBasicBlock(const typename InstructionStreamType::Ref&, unsigned blockIndex);
    inline BytecodeBasicBlock(SpecialBlockType, unsigned blockIndex);
    BytecodeBasicBlock(BytecodeBasicBlock<OpcodeTraits>&&) = default;


    bool isEntryBlock() { return !m_leaderOffset && !m_totalLength; }
    bool isExitBlock() { return m_leaderOffset == UINT_MAX && m_totalLength == UINT_MAX; }

    unsigned leaderOffset() const { return m_leaderOffset; }
    unsigned totalLength() const { return m_totalLength; }

    const Vector<uint8_t>& delta() const { return m_delta; }
    const Vector<unsigned>& successors() const { return m_successors; }

    FastBitVector& in() { return m_in; }
    FastBitVector& out() { return m_out; }

    unsigned index() const { return m_index; }

    explicit operator bool() const { return true; }

private:
    // Only called from BytecodeGraph.
    static BasicBlockVector compute(CodeBlock*, const InstructionStreamType& instructions);
    static BasicBlockVector compute(UnlinkedCodeBlockGenerator*, const InstructionStreamType& instructions);
    template<typename Block> static BasicBlockVector computeImpl(Block* codeBlock, const InstructionStreamType& instructions);
    void shrinkToFit();

    void addSuccessor(BytecodeBasicBlock<OpcodeTraits>& block)
    {
        if (!m_successors.contains(block.index()))
            m_successors.append(block.index());
    }

    inline void addLength(unsigned);

    typename InstructionStreamType::Offset m_leaderOffset;
    unsigned m_totalLength;
    unsigned m_index;

    Vector<uint8_t> m_delta;
    Vector<unsigned> m_successors;

    FastBitVector m_in;
    FastBitVector m_out;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<typename OpcodeTraits>, BytecodeBasicBlock<OpcodeTraits>);

using JSBytecodeBasicBlock = BytecodeBasicBlock<JSOpcodeTraits>;
using WasmBytecodeBasicBlock = BytecodeBasicBlock<WasmOpcodeTraits>;

} // namespace JSC
