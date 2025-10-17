/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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

#include "BytecodeGeneratorBase.h"

#include "RegisterID.h"
#include "StackAlignment.h"

namespace JSC {

template<typename T>
static inline void shrinkToFit(T& segmentedVector)
{
    while (segmentedVector.size() && !segmentedVector.last().refCount())
        segmentedVector.removeLast();
}

template<typename Traits>
BytecodeGeneratorBase<Traits>::BytecodeGeneratorBase(typename Traits::CodeBlock codeBlock, uint32_t virtualRegisterCountForCalleeSaves)
    : m_codeBlock(WTFMove(codeBlock))
{
    allocateCalleeSaveSpace(virtualRegisterCountForCalleeSaves);
}

template<typename Traits>
Ref<GenericLabel<Traits>> BytecodeGeneratorBase<Traits>::newLabel()
{
    shrinkToFit(m_labels);

    // Allocate new label ID.
    m_labels.append();
    return m_labels.last();
}

template<typename Traits>
Ref<GenericLabel<Traits>> BytecodeGeneratorBase<Traits>::newEmittedLabel()
{
    auto label = newLabel();
    emitLabel(label.get());
    return label;
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::reclaimFreeRegisters()
{
    shrinkToFit(m_calleeLocals);
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::emitLabel(GenericLabel<Traits>& label)
{
    unsigned newLabelIndex = m_writer.position();
    label.setLocation(*this, newLabelIndex);

    if (m_codeBlock->numberOfJumpTargets()) {
        unsigned lastLabelIndex = m_codeBlock->lastJumpTarget();
        ASSERT(lastLabelIndex <= newLabelIndex);
        if (newLabelIndex == lastLabelIndex) {
            // Peephole optimizations have already been disabled by emitting the last label
            return;
        }
    }

    m_codeBlock->addJumpTarget(newLabelIndex);

    m_lastOpcodeID = Traits::opcodeForDisablingOptimizations;
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::recordOpcode(typename Traits::OpcodeID opcodeID)
{
    ASSERT(m_lastOpcodeID == Traits::opcodeForDisablingOptimizations || (m_lastOpcodeID == m_lastInstruction->opcodeID() && m_writer.position() == m_lastInstruction.offset() + m_lastInstruction->size()));
    m_lastInstruction = m_writer.ref();
    m_lastOpcodeID = opcodeID;
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::alignWideOpcode16()
{
#if CPU(NEEDS_ALIGNED_ACCESS)
    static_assert(Traits::OpcodeTraits::maxOpcodeIDWidth == OpcodeSize::Narrow);
    size_t opcodeSize = 1;
    size_t prefixAndOpcodeSize = opcodeSize + PaddingBySize<OpcodeSize::Wide16>::value;
    while ((m_writer.position() + prefixAndOpcodeSize) % OpcodeSize::Wide16)
        Traits::OpNop::template emit<OpcodeSize::Narrow>(this);
#endif
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::alignWideOpcode32()
{
#if CPU(NEEDS_ALIGNED_ACCESS)
    static_assert(Traits::OpcodeTraits::maxOpcodeIDWidth == OpcodeSize::Narrow);
    size_t opcodeSize = 1;
    size_t prefixAndOpcodeSize = opcodeSize + PaddingBySize<OpcodeSize::Wide32>::value;
    while ((m_writer.position() + prefixAndOpcodeSize) % OpcodeSize::Wide32)
        Traits::OpNop::template emit<OpcodeSize::Narrow>(this);
#endif
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::write(uint8_t b)
{
    m_writer.write(b);
}


template<typename Traits>
void BytecodeGeneratorBase<Traits>::write(uint16_t h)
{
    m_writer.write(h);
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::write(uint32_t i)
{
    m_writer.write(i);
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::write(int8_t b)
{
    m_writer.write(static_cast<uint8_t>(b));
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::write(int16_t h)
{
    m_writer.write(static_cast<uint16_t>(h));
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::write(int32_t i)
{
    m_writer.write(static_cast<uint32_t>(i));
}

template<typename Traits>
RegisterID* BytecodeGeneratorBase<Traits>::newRegister()
{
    m_calleeLocals.append(virtualRegisterForLocal(m_calleeLocals.size()));
    size_t numCalleeLocals = std::max<size_t>(m_codeBlock->numCalleeLocals(), m_calleeLocals.size());
    numCalleeLocals = WTF::roundUpToMultipleOf(stackAlignmentRegisters(), numCalleeLocals);
    m_codeBlock->setNumCalleeLocals(static_cast<unsigned>(numCalleeLocals));
    RELEASE_ASSERT(numCalleeLocals == m_codeBlock->numCalleeLocals());
    return &m_calleeLocals.last();
}

template<typename Traits>
RegisterID* BytecodeGeneratorBase<Traits>::newTemporary()
{
    reclaimFreeRegisters();

    RegisterID* result = newRegister();
    result->setTemporary();
    return result;
}

template<typename Traits>
template<typename Functor>
void BytecodeGeneratorBase<Traits>::newTemporaries(size_t count, const Functor& func)
{
    reclaimFreeRegisters();
    for (size_t index = 0; index < count; ++index) {
        RegisterID* result = newRegister();
        result->setTemporary();
        func(result);
    }
}

// Adds an anonymous local var slot. To give this slot a name, add it to symbolTable().
template<typename Traits>
RegisterID* BytecodeGeneratorBase<Traits>::addVar()
{
    int numVars = m_codeBlock->numVars();
    m_codeBlock->setNumVars(numVars + 1);
    RegisterID* result = newRegister();
    ASSERT(VirtualRegister(result->index()).toLocal() == numVars);
    result->ref(); // We should never free this slot.
    return result;
}

template<typename Traits>
void BytecodeGeneratorBase<Traits>::allocateCalleeSaveSpace(uint32_t virtualRegisterCountForCalleeSaves)
{
    for (size_t i = 0; i < virtualRegisterCountForCalleeSaves; i++)
        addVar();
}

} // namespace JSC
