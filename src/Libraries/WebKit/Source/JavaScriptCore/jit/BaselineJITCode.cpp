/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 31, 2025.
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
#include "BaselineJITCode.h"

#if ENABLE(JIT)

#include "JITMathIC.h"
#include "JumpTable.h"
#include "PCToCodeOriginMap.h"
#include "StructureStubInfo.h"

namespace JSC {

JITAddIC* MathICHolder::addJITAddIC(BinaryArithProfile* arithProfile) { return m_addICs.add(arithProfile); }
JITMulIC* MathICHolder::addJITMulIC(BinaryArithProfile* arithProfile) { return m_mulICs.add(arithProfile); }
JITSubIC* MathICHolder::addJITSubIC(BinaryArithProfile* arithProfile) { return m_subICs.add(arithProfile); }
JITNegIC* MathICHolder::addJITNegIC(UnaryArithProfile* arithProfile) { return m_negICs.add(arithProfile); }

void MathICHolder::adoptMathICs(MathICHolder& other)
{
    m_addICs = WTFMove(other.m_addICs);
    m_mulICs = WTFMove(other.m_mulICs);
    m_negICs = WTFMove(other.m_negICs);
    m_subICs = WTFMove(other.m_subICs);
}

BaselineJITCode::BaselineJITCode(CodeRef<JSEntryPtrTag> code, CodePtr<JSEntryPtrTag> withArityCheck)
    : DirectJITCode(WTFMove(code), withArityCheck, JITType::BaselineJIT)
    , MathICHolder()
{ }

BaselineJITCode::~BaselineJITCode() = default;

CodeLocationLabel<JSInternalPtrTag> BaselineJITCode::getCallLinkDoneLocationForBytecodeIndex(BytecodeIndex bytecodeIndex) const
{
    auto* result = binarySearch<const BaselineUnlinkedCallLinkInfo, BytecodeIndex>(m_unlinkedCalls.span().data(), m_unlinkedCalls.size(), bytecodeIndex,
        [](const auto& value) {
            return value->bytecodeIndex;
        });
    if (!result)
        return { };
    return result->doneLocation;
}

BaselineJITData::BaselineJITData(unsigned stubInfoSize, unsigned poolSize, CodeBlock* codeBlock)
    : Base(stubInfoSize, poolSize)
    , m_globalObject(codeBlock->globalObject())
    , m_stackOffset(codeBlock->stackPointerOffset() * sizeof(Register))
{
}

} // namespace JSC

#endif // ENABLE(JIT)
