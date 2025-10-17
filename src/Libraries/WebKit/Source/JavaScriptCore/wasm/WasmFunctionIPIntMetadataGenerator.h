/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 17, 2022.
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

#include <wtf/Compiler.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

#if ENABLE(WEBASSEMBLY)

#include "BytecodeConventions.h"
#include "HandlerInfo.h"
#include "InstructionStream.h"
#include "MacroAssemblerCodeRef.h"
#include "SIMDInfo.h"
#include "WasmHandlerInfo.h"
#include "WasmIPIntGenerator.h"
#include "WasmIPIntTierUpCounter.h"
#include "WasmOps.h"
#include <wtf/BitVector.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC {

class JITCode;

template <typename Traits>
class BytecodeGeneratorBase;

namespace Wasm {

class IPIntCallee;
class TypeDefinition;
struct IPIntGeneratorTraits;
struct JumpTableEntry;

#define WRITE_TO_METADATA(dst, src, type) \
    do { \
        type tmp = src; \
        memcpy(dst, &tmp, sizeof(type)); \
    } while (false)

class FunctionIPIntMetadataGenerator {
    WTF_MAKE_TZONE_ALLOCATED(FunctionIPIntMetadataGenerator);
    WTF_MAKE_NONCOPYABLE(FunctionIPIntMetadataGenerator);

    friend class IPIntGenerator;
    friend class IPIntCallee;

public:
    FunctionIPIntMetadataGenerator(FunctionCodeIndex functionIndex, std::span<const uint8_t> bytecode)
        : m_functionIndex(functionIndex)
        , m_bytecode(bytecode)
    {
    }

    FunctionCodeIndex functionIndex() const { return m_functionIndex; }
    const BitVector& tailCallSuccessors() const { return m_tailCallSuccessors; }
    bool tailCallClobbersInstance() const { return m_tailCallClobbersInstance ; }
    void setTailCall(uint32_t, bool);
    void setTailCallClobbersInstance() { m_tailCallClobbersInstance = true; }

    const uint8_t* getBytecode() const { return m_bytecode.data(); }
    const uint8_t* getMetadata() const { return m_metadata.data(); }

    UncheckedKeyHashMap<IPIntPC, IPIntTierUpCounter::OSREntryData>& tierUpCounter() { return m_tierUpCounter; }

    unsigned addSignature(const TypeDefinition&);

private:

    inline void addBlankSpace(size_t);
    template <typename T> inline void addBlankSpace() { addBlankSpace(sizeof(T)); };

    template <typename T> inline void appendMetadata(T t)
    {
        auto size = m_metadata.size();
        addBlankSpace<T>();
        WRITE_TO_METADATA(m_metadata.data() + size, t, T);
    };

    void addLength(size_t length);
    void addLEB128ConstantInt32AndLength(uint32_t value, size_t length);
    void addLEB128ConstantAndLengthForType(Type, uint64_t value, size_t length);
    void addLEB128V128Constant(v128_t value, size_t length);
    void addReturnData(const FunctionSignature&);

    FunctionCodeIndex m_functionIndex;
    bool m_tailCallClobbersInstance { false };
    BitVector m_tailCallSuccessors;

    std::span<const uint8_t> m_bytecode;
    Vector<uint8_t> m_metadata { };
    Vector<uint8_t, 8> m_uINTBytecode { };
    unsigned m_highestReturnStackOffset;

    uint32_t m_bytecodeOffset { 0 };
    unsigned m_maxFrameSizeInV128 { 0 };
    unsigned m_numLocals { 0 };
    unsigned m_numAlignedRethrowSlots { 0 };
    unsigned m_numArguments { 0 };
    unsigned m_numArgumentsOnStack { 0 };
    unsigned m_nonArgLocalOffset { 0 };
    Vector<uint8_t, 16> m_argumINTBytecode { };

    Vector<const TypeDefinition*> m_signatures;
    UncheckedKeyHashMap<IPIntPC, IPIntTierUpCounter::OSREntryData> m_tierUpCounter;
    Vector<UnlinkedHandlerInfo> m_exceptionHandlers;
};

void FunctionIPIntMetadataGenerator::addBlankSpace(size_t size)
{
    m_metadata.grow(m_metadata.size() + size);
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY)

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
