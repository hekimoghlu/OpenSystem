/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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

#include "CallLinkInfo.h"
#include "JITCode.h"
#include "JITCodeMap.h"
#include "StructureStubInfo.h"
#include <wtf/ButterflyArray.h>
#include <wtf/CompactPointerTuple.h>

#if ENABLE(JIT)

namespace JSC {

class BinaryArithProfile;
class UnaryArithProfile;
struct BaselineUnlinkedStructureStubInfo;
struct SimpleJumpTable;
struct StringJumpTable;

class MathICHolder {
public:
    void adoptMathICs(MathICHolder& other);
    JITAddIC* addJITAddIC(BinaryArithProfile*);
    JITMulIC* addJITMulIC(BinaryArithProfile*);
    JITSubIC* addJITSubIC(BinaryArithProfile*);
    JITNegIC* addJITNegIC(UnaryArithProfile*);

private:
    Bag<JITAddIC> m_addICs;
    Bag<JITMulIC> m_mulICs;
    Bag<JITNegIC> m_negICs;
    Bag<JITSubIC> m_subICs;
};

class JITConstantPool {
    WTF_MAKE_NONCOPYABLE(JITConstantPool);
public:
    using Constant = unsigned;

    enum class Type : uint8_t {
        FunctionDecl,
        FunctionExpr,
    };

    using Value = JITConstant<Type>;

    JITConstantPool() = default;
    JITConstantPool(JITConstantPool&&) = default;
    JITConstantPool& operator=(JITConstantPool&&) = default;

    JITConstantPool(Vector<Value>&& constants)
        : m_constants(WTFMove(constants))
    {
    }

    size_t size() const { return m_constants.size(); }
    Value at(size_t i) const { return m_constants[i]; }

private:
    FixedVector<Value> m_constants;
};


class BaselineJITCode : public DirectJITCode, public MathICHolder {
public:
    BaselineJITCode(CodeRef<JSEntryPtrTag>, CodePtr<JSEntryPtrTag> withArityCheck);
    ~BaselineJITCode() override;
    PCToCodeOriginMap* pcToCodeOriginMap() override { return m_pcToCodeOriginMap.get(); }

    CodeLocationLabel<JSInternalPtrTag> getCallLinkDoneLocationForBytecodeIndex(BytecodeIndex) const;

    FixedVector<BaselineUnlinkedCallLinkInfo> m_unlinkedCalls;
    FixedVector<BaselineUnlinkedStructureStubInfo> m_unlinkedStubInfos;
    FixedVector<SimpleJumpTable> m_switchJumpTables;
    FixedVector<StringJumpTable> m_stringSwitchJumpTables;
    JITCodeMap m_jitCodeMap;
    JITConstantPool m_constantPool;
    std::unique_ptr<PCToCodeOriginMap> m_pcToCodeOriginMap;
    bool m_isShareable { true };
};

class BaselineJITData final : public ButterflyArray<BaselineJITData, StructureStubInfo, void*> {
    friend class LLIntOffsetsExtractor;
public:
    using Base = ButterflyArray<BaselineJITData, StructureStubInfo, void*>;

    static std::unique_ptr<BaselineJITData> create(unsigned stubInfoSize, unsigned poolSize, CodeBlock* codeBlock)
    {
        return std::unique_ptr<BaselineJITData> { createImpl(stubInfoSize, poolSize, codeBlock) };
    }

    explicit BaselineJITData(unsigned poolSize, unsigned stubInfoSize, CodeBlock*);

    static constexpr ptrdiff_t offsetOfGlobalObject() { return OBJECT_OFFSETOF(BaselineJITData, m_globalObject); }
    static constexpr ptrdiff_t offsetOfStackOffset() { return OBJECT_OFFSETOF(BaselineJITData, m_stackOffset); }
    static constexpr ptrdiff_t offsetOfJITExecuteCounter() { return OBJECT_OFFSETOF(BaselineJITData, m_executeCounter) + OBJECT_OFFSETOF(BaselineExecutionCounter, m_counter); }
    static constexpr ptrdiff_t offsetOfJITExecutionActiveThreshold() { return OBJECT_OFFSETOF(BaselineJITData, m_executeCounter) + OBJECT_OFFSETOF(BaselineExecutionCounter, m_activeThreshold); }
    static constexpr ptrdiff_t offsetOfJITExecutionTotalCount() { return OBJECT_OFFSETOF(BaselineJITData, m_executeCounter) + OBJECT_OFFSETOF(BaselineExecutionCounter, m_totalCount); }

    StructureStubInfo& stubInfo(unsigned index)
    {
        auto span = stubInfos();
        return span[span.size() - index - 1];
    }

    auto stubInfos() -> decltype(leadingSpan())
    {
        return leadingSpan();
    }

    BaselineExecutionCounter& executeCounter() { return m_executeCounter; }
    const BaselineExecutionCounter& executeCounter() const { return m_executeCounter; }

    JSGlobalObject* m_globalObject { nullptr }; // This is not marked since owner CodeBlock will mark JSGlobalObject.
    intptr_t m_stackOffset { 0 };
    BaselineExecutionCounter m_executeCounter;
};

} // namespace JSC

#endif // ENABLE(JIT)
