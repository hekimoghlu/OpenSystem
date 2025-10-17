/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#include "JITRightShiftGenerator.h"

#if ENABLE(JIT)

namespace JSC {

void JITRightShiftGenerator::generateFastPath(CCallHelpers& jit)
{
    ASSERT(m_scratchGPR != InvalidGPRReg);
    ASSERT(m_scratchGPR != m_left.payloadGPR());
    ASSERT(m_scratchGPR != m_right.payloadGPR());
#if USE(JSVALUE32_64)
    ASSERT(m_scratchGPR != m_left.tagGPR());
    ASSERT(m_scratchGPR != m_right.tagGPR());
#endif

    ASSERT(!m_leftOperand.isConstInt32() || !m_rightOperand.isConstInt32());

    m_didEmitFastPath = true;

    if (m_rightOperand.isConstInt32()) {
        // Try to do (intVar >> intConstant).
        CCallHelpers::Jump notInt = jit.branchIfNotInt32(m_left);

        jit.moveValueRegs(m_left, m_result);
        int32_t shiftAmount = m_rightOperand.asConstInt32() & 0x1f;
        if (shiftAmount) {
            if (m_shiftType == SignedShift)
                jit.rshift32(CCallHelpers::Imm32(shiftAmount), m_result.payloadGPR());
            else
                jit.urshift32(CCallHelpers::Imm32(shiftAmount), m_result.payloadGPR());
#if USE(JSVALUE64)
            jit.or64(GPRInfo::numberTagRegister, m_result.payloadGPR());
#endif
        }

        if (jit.supportsFloatingPointTruncate()) {
            m_endJumpList.append(jit.jump()); // Terminate the above case before emitting more code.

            // Try to do (doubleVar >> intConstant).
            notInt.link(&jit);

            m_slowPathJumpList.append(jit.branchIfNotNumber(m_left, m_scratchGPR));

            jit.unboxDoubleNonDestructive(m_left, m_leftFPR, m_scratchGPR);
#if CPU(ARM64)
            if (MacroAssemblerARM64::supportsDoubleToInt32ConversionUsingJavaScriptSemantics())
                jit.convertDoubleToInt32UsingJavaScriptSemantics(m_leftFPR, m_scratchGPR);
            else
#endif
            {
                m_slowPathJumpList.append(jit.branchTruncateDoubleToInt32(m_leftFPR, m_scratchGPR));
            }

            if (shiftAmount) {
                if (m_shiftType == SignedShift)
                    jit.rshift32(CCallHelpers::Imm32(shiftAmount), m_scratchGPR);
                else
                    jit.urshift32(CCallHelpers::Imm32(shiftAmount), m_scratchGPR);
            }
            jit.boxInt32(m_scratchGPR, m_result);

        } else
            m_slowPathJumpList.append(notInt);

    } else {
        // Try to do (intConstant >> intVar) or (intVar >> intVar).
        m_slowPathJumpList.append(jit.branchIfNotInt32(m_right));

        GPRReg rightOperandGPR = m_right.payloadGPR();
        if (rightOperandGPR == m_result.payloadGPR())
            rightOperandGPR = m_scratchGPR;

        CCallHelpers::Jump leftNotInt;
        if (m_leftOperand.isConstInt32()) {
            jit.move(m_right.payloadGPR(), rightOperandGPR);
#if USE(JSVALUE32_64)
            jit.move(m_right.tagGPR(), m_result.tagGPR());
#endif
            jit.move(CCallHelpers::Imm32(m_leftOperand.asConstInt32()), m_result.payloadGPR());
        } else {
            leftNotInt = jit.branchIfNotInt32(m_left);
            jit.move(m_right.payloadGPR(), rightOperandGPR);
            jit.moveValueRegs(m_left, m_result);
        }

        if (m_shiftType == SignedShift)
            jit.rshift32(rightOperandGPR, m_result.payloadGPR());
        else
            jit.urshift32(rightOperandGPR, m_result.payloadGPR());
#if USE(JSVALUE64)
        jit.or64(GPRInfo::numberTagRegister, m_result.payloadGPR());
#endif
        if (m_leftOperand.isConstInt32())
            return;

        if (jit.supportsFloatingPointTruncate()) {
            m_endJumpList.append(jit.jump()); // Terminate the above case before emitting more code.

            // Try to do (doubleVar >> intVar).
            leftNotInt.link(&jit);

            m_slowPathJumpList.append(jit.branchIfNotNumber(m_left, m_scratchGPR));
            jit.unboxDoubleNonDestructive(m_left, m_leftFPR, m_scratchGPR);
#if CPU(ARM64)
            if (MacroAssemblerARM64::supportsDoubleToInt32ConversionUsingJavaScriptSemantics())
                jit.convertDoubleToInt32UsingJavaScriptSemantics(m_leftFPR, m_scratchGPR);
            else
#endif
            {
                m_slowPathJumpList.append(jit.branchTruncateDoubleToInt32(m_leftFPR, m_scratchGPR));
            }

            if (m_shiftType == SignedShift)
                jit.rshift32(m_right.payloadGPR(), m_scratchGPR);
            else
                jit.urshift32(m_right.payloadGPR(), m_scratchGPR);
            jit.boxInt32(m_scratchGPR, m_result);

        } else
            m_slowPathJumpList.append(leftNotInt);
    }
}

} // namespace JSC

#endif // ENABLE(JIT)
