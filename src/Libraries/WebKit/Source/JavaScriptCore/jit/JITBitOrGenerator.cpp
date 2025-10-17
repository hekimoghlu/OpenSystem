/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#include "JITBitOrGenerator.h"

#if ENABLE(JIT)

namespace JSC {

void JITBitOrGenerator::generateFastPath(CCallHelpers& jit)
{
    ASSERT(!m_leftOperand.isConstInt32() || !m_rightOperand.isConstInt32());
    
    m_didEmitFastPath = true;
    
    if (m_leftOperand.isConstInt32() || m_rightOperand.isConstInt32()) {
        JSValueRegs var = m_leftOperand.isConstInt32() ? m_right : m_left;
        SnippetOperand& constOpr = m_leftOperand.isConstInt32() ? m_leftOperand : m_rightOperand;
        
        // Try to do intVar | intConstant.
        m_slowPathJumpList.append(jit.branchIfNotInt32(var));

        if (constOpr.asConstInt32()) {
#if USE(JSVALUE64)
#if CPU(ARM64)
            jit.or64(CCallHelpers::TrustedImm64(static_cast<uint64_t>(static_cast<uint32_t>(constOpr.asConstInt32()))), var.payloadGPR(), m_result.payloadGPR());
#else
            jit.moveValueRegs(var, m_result);
            jit.or32(CCallHelpers::Imm32(constOpr.asConstInt32()), m_result.payloadGPR());
            jit.or64(GPRInfo::numberTagRegister, m_result.payloadGPR());
#endif
#else
            jit.moveValueRegs(var, m_result);
            jit.or32(CCallHelpers::Imm32(constOpr.asConstInt32()), m_result.payloadGPR());
#endif
        } else
            jit.moveValueRegs(var, m_result);
        return;
    }

#if USE(JSVALUE64)
    if (m_leftOperand.definitelyIsBoolean() && m_rightOperand.definitelyIsBoolean()) {
        jit.or32(m_left.payloadGPR(), m_right.payloadGPR(), m_result.payloadGPR());
        jit.and32(CCallHelpers::TrustedImm32(1), m_result.payloadGPR());
        jit.or64(GPRInfo::numberTagRegister, m_result.payloadGPR());
        return;
    }
#endif

    ASSERT(!m_leftOperand.isConstInt32() && !m_rightOperand.isConstInt32());

    // Try to do intVar | intVar.
    m_slowPathJumpList.append(jit.branchIfNotInt32(m_left));
    m_slowPathJumpList.append(jit.branchIfNotInt32(m_right));

#if USE(JSVALUE64)
    jit.or64(m_right.payloadGPR(), m_left.payloadGPR(), m_result.payloadGPR());
#else
    jit.moveValueRegs(m_left, m_result);
    jit.or32(m_right.payloadGPR(), m_result.payloadGPR());
#endif
}

} // namespace JSC

#endif // ENABLE(JIT)
