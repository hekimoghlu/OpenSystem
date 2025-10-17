/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
#include "JITMulGenerator.h"

#if ENABLE(JIT)

#include "ArithProfile.h"
#include "JITMathIC.h"

namespace JSC {

JITMathICInlineResult JITMulGenerator::generateInline(CCallHelpers& jit, MathICGenerationState& state, const BinaryArithProfile* arithProfile)
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

    if (lhs.isOnlyNumber() && rhs.isOnlyNumber() && !m_leftOperand.isConst() && !m_rightOperand.isConst()) {
        if (!jit.supportsFloatingPoint())
            return JITMathICInlineResult::DontGenerate;

        ASSERT(m_left);
        ASSERT(m_right);
        if (!m_leftOperand.definitelyIsNumber())
            state.slowPathJumps.append(jit.branchIfNotNumber(m_left, m_scratchGPR));
        if (!m_rightOperand.definitelyIsNumber())
            state.slowPathJumps.append(jit.branchIfNotNumber(m_right, m_scratchGPR));
        state.slowPathJumps.append(jit.branchIfInt32(m_left));
        state.slowPathJumps.append(jit.branchIfInt32(m_right));
        jit.unboxDoubleNonDestructive(m_left, m_leftFPR, m_scratchGPR);
        jit.unboxDoubleNonDestructive(m_right, m_rightFPR, m_scratchGPR);
        jit.mulDouble(m_rightFPR, m_leftFPR);
        jit.boxDouble(m_leftFPR, m_result);

        return JITMathICInlineResult::GeneratedFastPath;
    }

    if ((lhs.isOnlyInt32() || m_leftOperand.isPositiveConstInt32()) && (rhs.isOnlyInt32() || m_rightOperand.isPositiveConstInt32())) {
        ASSERT(!m_leftOperand.isPositiveConstInt32() || !m_rightOperand.isPositiveConstInt32());
        if (!m_leftOperand.isPositiveConstInt32())
            state.slowPathJumps.append(jit.branchIfNotInt32(m_left));
        if (!m_rightOperand.isPositiveConstInt32())
            state.slowPathJumps.append(jit.branchIfNotInt32(m_right));

        if (m_leftOperand.isPositiveConstInt32() || m_rightOperand.isPositiveConstInt32()) {
            JSValueRegs var = m_leftOperand.isPositiveConstInt32() ? m_right : m_left;
            int32_t constValue = m_leftOperand.isPositiveConstInt32() ? m_leftOperand.asConstInt32() : m_rightOperand.asConstInt32();
            state.slowPathJumps.append(jit.branchMul32(CCallHelpers::Overflow, var.payloadGPR(), CCallHelpers::Imm32(constValue), m_scratchGPR));
        } else {
            state.slowPathJumps.append(jit.branchMul32(CCallHelpers::Overflow, m_right.payloadGPR(), m_left.payloadGPR(), m_scratchGPR));
            state.slowPathJumps.append(jit.branchTest32(CCallHelpers::Zero, m_scratchGPR)); // Go slow if potential negative zero.
        }
        jit.boxInt32(m_scratchGPR, m_result);

        return JITMathICInlineResult::GeneratedFastPath;
    }

    return JITMathICInlineResult::GenerateFullSnippet;
}

bool JITMulGenerator::generateFastPath(CCallHelpers& jit, CCallHelpers::JumpList& endJumpList, CCallHelpers::JumpList& slowPathJumpList, const BinaryArithProfile* arithProfile, bool shouldEmitProfiling)
{
    ASSERT(m_scratchGPR != InvalidGPRReg);
    ASSERT(m_scratchGPR != m_left.payloadGPR());
    ASSERT(m_scratchGPR != m_right.payloadGPR());
#if USE(JSVALUE64)
    ASSERT(m_scratchGPR != m_result.payloadGPR());
#else
    ASSERT(m_scratchGPR != m_left.tagGPR());
    ASSERT(m_scratchGPR != m_right.tagGPR());
#endif

    ASSERT(!m_leftOperand.isPositiveConstInt32() || !m_rightOperand.isPositiveConstInt32());

    if (!m_leftOperand.mightBeNumber() || !m_rightOperand.mightBeNumber())
        return false;

    if (m_leftOperand.isPositiveConstInt32() || m_rightOperand.isPositiveConstInt32()) {
        JSValueRegs var = m_leftOperand.isPositiveConstInt32() ? m_right : m_left;
        SnippetOperand& varOpr = m_leftOperand.isPositiveConstInt32() ? m_rightOperand : m_leftOperand;
        SnippetOperand& constOpr = m_leftOperand.isPositiveConstInt32() ? m_leftOperand : m_rightOperand;

        // Try to do intVar * intConstant.
        CCallHelpers::Jump notInt32 = jit.branchIfNotInt32(var);

        GPRReg multiplyResultGPR = m_result.payloadGPR();
        if (multiplyResultGPR == var.payloadGPR())
            multiplyResultGPR = m_scratchGPR;

        slowPathJumpList.append(jit.branchMul32(CCallHelpers::Overflow, var.payloadGPR(), CCallHelpers::Imm32(constOpr.asConstInt32()), multiplyResultGPR));

        jit.boxInt32(multiplyResultGPR, m_result);
        endJumpList.append(jit.jump());

        if (!jit.supportsFloatingPoint()) {
            slowPathJumpList.append(notInt32);
            return true;
        }

        // Try to do doubleVar * double(intConstant).
        notInt32.link(&jit);
        if (!varOpr.definitelyIsNumber())
            slowPathJumpList.append(jit.branchIfNotNumber(var, m_scratchGPR));

        jit.unboxDoubleNonDestructive(var, m_leftFPR, m_scratchGPR);

        jit.move(CCallHelpers::Imm32(constOpr.asConstInt32()), m_scratchGPR);
        jit.convertInt32ToDouble(m_scratchGPR, m_rightFPR);

        // Fall thru to doubleVar * doubleVar.

    } else {
        ASSERT(!m_leftOperand.isPositiveConstInt32() && !m_rightOperand.isPositiveConstInt32());

        CCallHelpers::Jump leftNotInt;
        CCallHelpers::Jump rightNotInt;

        // Try to do intVar * intVar.
        leftNotInt = jit.branchIfNotInt32(m_left);
        rightNotInt = jit.branchIfNotInt32(m_right);

        slowPathJumpList.append(jit.branchMul32(CCallHelpers::Overflow, m_right.payloadGPR(), m_left.payloadGPR(), m_scratchGPR));
        slowPathJumpList.append(jit.branchTest32(CCallHelpers::Zero, m_scratchGPR)); // Go slow if potential negative zero.

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

        // Fall thru to doubleVar * doubleVar.
    }

    // Do doubleVar * doubleVar.
    jit.mulDouble(m_rightFPR, m_leftFPR);

    if (!arithProfile || !shouldEmitProfiling)
        jit.boxDouble(m_leftFPR, m_result);
    else {
        // The Int52 overflow check below intentionally omits 1ll << 51 as a valid negative Int52 value.
        // Therefore, we will get a false positive if the result is that value. This is intentionally
        // done to simplify the checking algorithm.

        const int64_t negativeZeroBits = 1ll << 63;
#if USE(JSVALUE64)
        jit.moveDoubleTo64(m_leftFPR, m_result.payloadGPR());

        CCallHelpers::Jump notNegativeZero = jit.branch64(CCallHelpers::NotEqual, m_result.payloadGPR(), CCallHelpers::TrustedImm64(negativeZeroBits));

        arithProfile->emitUnconditionalSet(jit, ObservedResults::NegZeroDouble);
        CCallHelpers::Jump done = jit.jump();

        notNegativeZero.link(&jit);
        arithProfile->emitUnconditionalSet(jit, ObservedResults::NonNegZeroDouble);

        jit.move(m_result.payloadGPR(), m_scratchGPR);
        jit.urshiftPtr(CCallHelpers::Imm32(52), m_scratchGPR);
        jit.and32(CCallHelpers::Imm32(0x7ff), m_scratchGPR);
        CCallHelpers::Jump noInt52Overflow = jit.branch32(CCallHelpers::LessThanOrEqual, m_scratchGPR, CCallHelpers::TrustedImm32(0x431));

        arithProfile->emitUnconditionalSet(jit, ObservedResults::Int52Overflow);
        noInt52Overflow.link(&jit);

        done.link(&jit);
        jit.sub64(GPRInfo::numberTagRegister, m_result.payloadGPR()); // Box the double.
#else
        jit.boxDouble(m_leftFPR, m_result);
        CCallHelpers::JumpList notNegativeZero;
        notNegativeZero.append(jit.branch32(CCallHelpers::NotEqual, m_result.payloadGPR(), CCallHelpers::TrustedImm32(0)));
        notNegativeZero.append(jit.branch32(CCallHelpers::NotEqual, m_result.tagGPR(), CCallHelpers::TrustedImm32(negativeZeroBits >> 32)));

        arithProfile->emitUnconditionalSet(jit, ObservedResults::NegZeroDouble);
        CCallHelpers::Jump done = jit.jump();

        notNegativeZero.link(&jit);
        arithProfile->emitUnconditionalSet(jit, ObservedResults::NonNegZeroDouble);

        jit.move(m_result.tagGPR(), m_scratchGPR);
        jit.urshiftPtr(CCallHelpers::Imm32(52 - 32), m_scratchGPR);
        jit.and32(CCallHelpers::Imm32(0x7ff), m_scratchGPR);
        CCallHelpers::Jump noInt52Overflow = jit.branch32(CCallHelpers::LessThanOrEqual, m_scratchGPR, CCallHelpers::TrustedImm32(0x431));

        arithProfile->emitUnconditionalSet(jit, ObservedResults::Int52Overflow);

        endJumpList.append(noInt52Overflow);
        if (m_scratchGPR == m_result.tagGPR() || m_scratchGPR == m_result.payloadGPR())
            jit.boxDouble(m_leftFPR, m_result);

        endJumpList.append(done);
#endif
    }

    return true;
}

} // namespace JSC

#endif // ENABLE(JIT)
