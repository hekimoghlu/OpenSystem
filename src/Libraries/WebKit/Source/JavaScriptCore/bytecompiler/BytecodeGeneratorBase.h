/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 4, 2022.
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
#include <wtf/SegmentedVector.h>

namespace JSC {

class RegisterID;

template<typename BytecodeGenerator>
class GenericBoundLabel;

template<typename BytecodeGenerator>
class GenericLabel;

template<typename Traits>
class BytecodeGeneratorBase {
    template<typename BytecodeGenerator>
    friend class GenericBoundLabel;

    template<typename BytecodeGenerator>
    friend class GenericLabel;

    using InstructionStreamWriterType = InstructionStreamWriter<typename Traits::InstructionType>;
    using InstructionStreamType = InstructionStream<typename Traits::InstructionType>;

public:
    BytecodeGeneratorBase(typename Traits::CodeBlock, uint32_t virtualRegisterCountForCalleeSaves);

    void allocateCalleeSaveSpace(uint32_t virtualRegisterCountForCalleeSaves);

    Ref<GenericLabel<Traits>> newLabel();
    Ref<GenericLabel<Traits>> newEmittedLabel();
    RegisterID* newRegister();
    RegisterID* addVar();

    // Returns the next available temporary register. Registers returned by
    // newTemporary require a modified form of reference counting: any
    // register with a refcount of 0 is considered "available", meaning that
    // the next instruction may overwrite it.
    RegisterID* newTemporary();
    template<typename Functor>
    void newTemporaries(size_t count, const Functor&);

    void emitLabel(GenericLabel<Traits>&);
    void recordOpcode(typename Traits::OpcodeID);
    void alignWideOpcode16();
    void alignWideOpcode32();

    void write(uint8_t);
    void write(uint16_t);
    void write(uint32_t);
    void write(int8_t);
    void write(int16_t);
    void write(int32_t);

protected:
    void reclaimFreeRegisters();

    InstructionStreamWriterType m_writer;
    typename Traits::CodeBlock m_codeBlock;

    bool m_outOfMemoryDuringConstruction { false };
    typename Traits::OpcodeID m_lastOpcodeID = Traits::opcodeForDisablingOptimizations;
    typename InstructionStreamType::MutableRef m_lastInstruction { m_writer.ref() };

    SegmentedVector<GenericLabel<Traits>, 32> m_labels;
    SegmentedVector<RegisterID, 32> m_calleeLocals;
};

} // namespace JSC
