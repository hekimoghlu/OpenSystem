/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 27, 2022.
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
#include "JITNegGenerator.h"

#include "ArithProfile.h"

#if ENABLE(JIT)

namespace JSC {

JITMathICInlineResult JITNegGenerator::generateInline(CCallHelpers& jit, MathICGenerationState& state, const UnaryArithProfile* arithProfile)
{
    ASSERT(m_scratchGPR != InvalidGPRReg);
    ASSERT(m_scratchGPR != m_src.payloadGPR());
    ASSERT(m_scratchGPR != m_result.payloadGPR());
#if USE(JSVALUE32_64)
    ASSERT(m_scratchGPR != m_src.tagGPR());
    ASSERT(m_scratchGPR != m_result.tagGPR());
#endif

    // We default to speculating int32.
    ObservedType observedTypes = ObservedType().withInt32();
    if (arithProfile)
        observedTypes = arithProfile->argObservedType();
    ASSERT_WITH_MESSAGE(!observedTypes.isEmpty(), "We should not attempt to generate anything if we do not have a profile.");

    if (observedTypes.isOnlyNonNumber())
        return JITMathICInlineResult::DontGenerate;

    if (observedTypes.isOnlyInt32()) {
        jit.moveValueRegs(m_src, m_result);
        state.slowPathJumps.append(jit.branchIfNotInt32(m_src));
        state.slowPathJumps.append(jit.branchTest32(CCallHelpers::Zero, m_src.payloadGPR(), CCallHelpers::TrustedImm32(0x7fffffff)));
        jit.neg32(m_result.payloadGPR());
#if USE(JSVALUE64)
        jit.boxInt32(m_result.payloadGPR(), m_result);
#endif

        return JITMathICInlineResult::GeneratedFastPath;
    }
    if (observedTypes.isOnlyNumber()) {
        state.slowPathJumps.append(jit.branchIfInt32(m_src));
        state.slowPathJumps.append(jit.branchIfNotNumber(m_src, m_scratchGPR));
#if USE(JSVALUE64)
        if (m_src.payloadGPR() != m_result.payloadGPR()) {
            jit.move(CCallHelpers::TrustedImm64(static_cast<int64_t>(1ull << 63)), m_result.payloadGPR());
            jit.xor64(m_src.payloadGPR(), m_result.payloadGPR());
        } else {
            jit.move(CCallHelpers::TrustedImm64(static_cast<int64_t>(1ull << 63)), m_scratchGPR);
            jit.xor64(m_scratchGPR, m_result.payloadGPR());
        }
#else
        jit.moveValueRegs(m_src, m_result);
        jit.xor32(CCallHelpers::TrustedImm32(1 << 31), m_result.tagGPR());
#endif
        return JITMathICInlineResult::GeneratedFastPath;
    }
    return JITMathICInlineResult::GenerateFullSnippet;
}

bool JITNegGenerator::generateFastPath(CCallHelpers& jit, CCallHelpers::JumpList& endJumpList, CCallHelpers::JumpList& slowPathJumpList, const UnaryArithProfile* arithProfile, bool shouldEmitProfiling)
{
    ASSERT(m_scratchGPR != m_src.payloadGPR());
    ASSERT(m_scratchGPR != m_result.payloadGPR());
    ASSERT(m_scratchGPR != InvalidGPRReg);
#if USE(JSVALUE32_64)
    ASSERT(m_scratchGPR != m_src.tagGPR());
    ASSERT(m_scratchGPR != m_result.tagGPR());
#endif

    jit.moveValueRegs(m_src, m_result);
    CCallHelpers::Jump srcNotInt = jit.branchIfNotInt32(m_src);

    // -0 should produce a double, and hence cannot be negated as an int.
    // The negative int32 0x80000000 doesn't have a positive int32 representation, and hence cannot be negated as an int.
    slowPathJumpList.append(jit.branchTest32(CCallHelpers::Zero, m_src.payloadGPR(), CCallHelpers::TrustedImm32(0x7fffffff)));

    jit.neg32(m_result.payloadGPR());
#if USE(JSVALUE64)
    jit.boxInt32(m_result.payloadGPR(), m_result);
#endif
    endJumpList.append(jit.jump());

    srcNotInt.link(&jit);
    slowPathJumpList.append(jit.branchIfNotNumber(m_src, m_scratchGPR));

    // For a double, all we need to do is to invert the sign bit.
#if USE(JSVALUE64)
    jit.move(CCallHelpers::TrustedImm64((int64_t)(1ull << 63)), m_scratchGPR);
    jit.xor64(m_scratchGPR, m_result.payloadGPR());
#else
    jit.xor32(CCallHelpers::TrustedImm32(1 << 31), m_result.tagGPR());
#endif
    // The flags of ArithNegate are basic in DFG.
    // We only need to know if we ever produced a number.
    if (shouldEmitProfiling && arithProfile && !arithProfile->argObservedType().sawNumber() && !arithProfile->didObserveDouble())
        arithProfile->emitSetDouble(jit, m_scratchGPR);
    return true;
}

} // namespace JSC

#endif // ENABLE(JIT)
