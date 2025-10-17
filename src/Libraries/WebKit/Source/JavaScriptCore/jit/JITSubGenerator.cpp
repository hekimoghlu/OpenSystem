/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 29, 2023.
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
#include "JITSubGenerator.h"

#include "ArithProfile.h"
#include "JITMathIC.h"

#if ENABLE(JIT)

namespace JSC {

JITMathICInlineResult JITSubGenerator::generateInline(CCallHelpers& jit, MathICGenerationState& state, const BinaryArithProfile* arithProfile)
{
    // We default to speculating int32.
    ObservedType lhs = ObservedType().withInt32();
    ObservedType rhs = ObservedType().withInt32();
    if (arithProfile) {
        lhs = arithProfile->lhsObservedType();
        rhs = arithProfile->rhsObservedType();
    }

    if (lhs.isOnlyNonNumber() && rhs.isOnlyNonNumber())
        return JITMathICInlineResult::DontGenerate;

    if (lhs.isOnlyNumber() && rhs.isOnlyNumber()) {
        if (!jit.supportsFloatingPoint())
            return JITMathICInlineResult::DontGenerate;

        if (!m_leftOperand.definitelyIsNumber())
            state.slowPathJumps.append(jit.branchIfNotNumber(m_left, m_scratchGPR));
        if (!m_rightOperand.definitelyIsNumber())
            state.slowPathJumps.append(jit.branchIfNotNumber(m_right, m_scratchGPR));
        state.slowPathJumps.append(jit.branchIfInt32(m_left));
        state.slowPathJumps.append(jit.branchIfInt32(m_right));
        jit.unboxDoubleNonDestructive(m_left, m_leftFPR, m_scratchGPR);
        jit.unboxDoubleNonDestructive(m_right, m_rightFPR, m_scratchGPR);
        jit.subDouble(m_rightFPR, m_leftFPR);
        jit.boxDouble(m_leftFPR, m_result);

        return JITMathICInlineResult::GeneratedFastPath;
    }
    if (lhs.isOnlyInt32() && rhs.isOnlyInt32()) {
        ASSERT(!m_leftOperand.isConstInt32() || !m_rightOperand.isConstInt32());
        state.slowPathJumps.append(jit.branchIfNotInt32(m_left));
        state.slowPathJumps.append(jit.branchIfNotInt32(m_right));

        jit.move(m_left.payloadGPR(), m_scratchGPR);
        state.slowPathJumps.append(jit.branchSub32(CCallHelpers::Overflow, m_right.payloadGPR(), m_scratchGPR));

        jit.boxInt32(m_scratchGPR, m_result);
        return JITMathICInlineResult::GeneratedFastPath;
    }

    return JITMathICInlineResult::GenerateFullSnippet;
}

bool JITSubGenerator::generateFastPath(CCallHelpers& jit, CCallHelpers::JumpList& endJumpList, CCallHelpers::JumpList& slowPathJumpList, const BinaryArithProfile* arithProfile, bool shouldEmitProfiling)
{
    ASSERT(m_scratchGPR != InvalidGPRReg);
    ASSERT(m_scratchGPR != m_left.payloadGPR());
    ASSERT(m_scratchGPR != m_right.payloadGPR());
#if USE(JSVALUE32_64)
    ASSERT(m_scratchGPR != m_left.tagGPR());
    ASSERT(m_scratchGPR != m_right.tagGPR());
#endif

    CCallHelpers::Jump leftNotInt = jit.branchIfNotInt32(m_left);
    CCallHelpers::Jump rightNotInt = jit.branchIfNotInt32(m_right);

    jit.move(m_left.payloadGPR(), m_scratchGPR);
    slowPathJumpList.append(jit.branchSub32(CCallHelpers::Overflow, m_right.payloadGPR(), m_scratchGPR));

    jit.boxInt32(m_scratchGPR, m_result);

    endJumpList.append(jit.jump());

    if (!jit.supportsFloatingPoint()) {
        slowPathJumpList.append(leftNotInt);
        slowPathJumpList.append(rightNotInt);
        return true;
    }

    leftNotInt.link(&jit);
    if (!m_leftOperand.definitelyIsNumber())
        slowPathJumpList.append(jit.branchIfNotNumber(m_left, m_scratchGPR));
    if (!m_rightOperand.definitelyIsNumber())
        slowPathJumpList.append(jit.branchIfNotNumber(m_right, m_scratchGPR));

    jit.unboxDoubleNonDestructive(m_left, m_leftFPR, m_scratchGPR);
    CCallHelpers::Jump rightIsDouble = jit.branchIfNotInt32(m_right);

    jit.convertInt32ToDouble(m_right.payloadGPR(), m_rightFPR);
    CCallHelpers::Jump rightWasInteger = jit.jump();

    rightNotInt.link(&jit);
    if (!m_rightOperand.definitelyIsNumber())
        slowPathJumpList.append(jit.branchIfNotNumber(m_right, m_scratchGPR));

    jit.convertInt32ToDouble(m_left.payloadGPR(), m_leftFPR);

    rightIsDouble.link(&jit);
    jit.unboxDoubleNonDestructive(m_right, m_rightFPR, m_scratchGPR);

    rightWasInteger.link(&jit);

    jit.subDouble(m_rightFPR, m_leftFPR);
    if (arithProfile && shouldEmitProfiling)
        arithProfile->emitSetDouble(jit, m_scratchGPR);

    jit.boxDouble(m_leftFPR, m_result);

    return true;
}

} // namespace JSC

#endif // ENABLE(JIT)
