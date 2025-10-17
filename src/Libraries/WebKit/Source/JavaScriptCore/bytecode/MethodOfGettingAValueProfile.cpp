/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 25, 2024.
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
#include "MethodOfGettingAValueProfile.h"

#if ENABLE(DFG_JIT)

#include "ArithProfile.h"
#include "CCallHelpers.h"
#include "CodeBlock.h"
#include "JSCJSValueInlines.h"

namespace JSC {

void MethodOfGettingAValueProfile::emitReportValue(CCallHelpers& jit, CodeBlock* optimizedCodeBlock, JSValueRegs regs, GPRReg tempGPR, TagRegistersMode mode) const
{
    if (m_kind == Kind::None)
        return;

    CodeBlock* baselineCodeBlock = optimizedCodeBlock->baselineAlternative();
    CodeBlock* profiledBlock = baselineCodeBlockForOriginAndBaselineCodeBlock(m_codeOrigin, baselineCodeBlock);
    switch (m_kind) {
    case Kind::None:
        RELEASE_ASSERT_NOT_REACHED();
        return;
        
    case Kind::LazyOperandValueProfile: {
        LazyOperandValueProfileKey key(m_codeOrigin.bytecodeIndex(), Operand::fromBits(m_rawOperand));
        
        LazyOperandValueProfile* profile = profiledBlock->lazyValueProfiles().addOperandValueProfile(key);
        jit.storeValue(regs, profile->specFailBucket(0));
        return;
    }
        
    case Kind::UnaryArithProfile: {
        if (UnaryArithProfile* result = profiledBlock->unaryArithProfileForBytecodeIndex(m_codeOrigin.bytecodeIndex()))
            result->emitObserveResult(jit, regs, tempGPR, mode);
        return;
    }

    case Kind::BinaryArithProfile: {
        if (BinaryArithProfile* result = profiledBlock->binaryArithProfileForBytecodeIndex(m_codeOrigin.bytecodeIndex()))
            result->emitObserveResult(jit, regs, tempGPR, mode);
        return;
    }

    case Kind::ArgumentValueProfile: {
        auto& valueProfile = profiledBlock->valueProfileForArgument(Operand::fromBits(m_rawOperand).toArgument());
        jit.storeValue(regs, valueProfile.specFailBucket(0));
        return;
    }

    case Kind::BytecodeValueProfile: {
        JSValue* bucket = profiledBlock->lazyValueProfiles().addSpeculationFailureValueProfile(m_codeOrigin.bytecodeIndex());
        jit.storeValue(regs, bucket);
        return;
    }
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace JSC

#endif // ENABLE(DFG_JIT)

