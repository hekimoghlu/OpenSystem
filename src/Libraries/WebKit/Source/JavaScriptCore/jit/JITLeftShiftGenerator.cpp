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
#include "config.h"
#include "JITLeftShiftGenerator.h"

#if ENABLE(JIT)

namespace JSC {

void JITLeftShiftGenerator::generateFastPath(CCallHelpers& jit)
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
        // Try to do (intVar << intConstant).
        m_slowPathJumpList.append(jit.branchIfNotInt32(m_left));

        jit.moveValueRegs(m_left, m_result);
        jit.lshift32(CCallHelpers::Imm32(m_rightOperand.asConstInt32()), m_result.payloadGPR());

    } else {
        // Try to do (intConstant << intVar) or (intVar << intVar).
        m_slowPathJumpList.append(jit.branchIfNotInt32(m_right));

        GPRReg rightOperandGPR = m_right.payloadGPR();
        if (rightOperandGPR == m_result.payloadGPR()) {
            jit.move(rightOperandGPR, m_scratchGPR);
            rightOperandGPR = m_scratchGPR;
        }

        if (m_leftOperand.isConstInt32()) {
#if USE(JSVALUE32_64)
            jit.move(m_right.tagGPR(), m_result.tagGPR());
#endif
            jit.move(CCallHelpers::Imm32(m_leftOperand.asConstInt32()), m_result.payloadGPR());
        } else {
            m_slowPathJumpList.append(jit.branchIfNotInt32(m_left));
            jit.moveValueRegs(m_left, m_result);
        }

        jit.lshift32(rightOperandGPR, m_result.payloadGPR());
    }

#if USE(JSVALUE64)
    jit.or64(GPRInfo::numberTagRegister, m_result.payloadGPR());
#endif
}

} // namespace JSC

#endif // ENABLE(JIT)
