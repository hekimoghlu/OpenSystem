/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
#include "B3StackmapGenerationParams.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirGenerationContext.h"
#include "B3Procedure.h"
#include "B3StackmapValue.h"

namespace JSC { namespace B3 {

const RegisterSetBuilder& StackmapGenerationParams::usedRegisters() const
{
    ASSERT(m_context.code->needsUsedRegisters());
    
    return m_value->m_usedRegisters;
}

RegisterSetBuilder StackmapGenerationParams::unavailableRegisters() const
{
    RegisterSetBuilder result = usedRegisters();
    
    RegisterSetBuilder unsavedCalleeSaves = RegisterSetBuilder::calleeSaveRegisters();
    ASSERT(!unsavedCalleeSaves.hasAnyWideRegisters());
    unsavedCalleeSaves.exclude(m_context.code->calleeSaveRegisters());

    result.merge(unsavedCalleeSaves);

    for (GPRReg gpr : m_gpScratch)
        result.remove(gpr);
    for (FPRReg fpr : m_fpScratch)
        result.remove(fpr);
    
    return result;
}

Vector<Box<MacroAssembler::Label>> StackmapGenerationParams::successorLabels() const
{
    RELEASE_ASSERT(m_context.indexInBlock == m_context.currentBlock->size() - 1);
    RELEASE_ASSERT(m_value->effects().terminal);
    
    Vector<Box<MacroAssembler::Label>> result(m_context.currentBlock->numSuccessors());
    for (unsigned i = m_context.currentBlock->numSuccessors(); i--;)
        result[i] = m_context.blockLabels[m_context.currentBlock->successorBlock(i)];
    return result;
}

bool StackmapGenerationParams::fallsThroughToSuccessor(unsigned successorIndex) const
{
    RELEASE_ASSERT(m_context.indexInBlock == m_context.currentBlock->size() - 1);
    RELEASE_ASSERT(m_value->effects().terminal);
    
    Air::BasicBlock* successor = m_context.currentBlock->successorBlock(successorIndex);
    Air::BasicBlock* nextBlock = m_context.code->findNextBlock(m_context.currentBlock);
    return successor == nextBlock;
}

Procedure& StackmapGenerationParams::proc() const
{
    return m_context.code->proc();
}

Air::Code& StackmapGenerationParams::code() const
{
    return proc().code();
}

StackmapGenerationParams::StackmapGenerationParams(
    StackmapValue* value, const Vector<ValueRep>& reps, Air::GenerationContext& context)
    : m_value(value)
    , m_reps(reps)
    , m_context(context)
{
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

