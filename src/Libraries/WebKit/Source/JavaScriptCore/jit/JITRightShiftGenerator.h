/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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

#if ENABLE(JIT)

#include "JITBitBinaryOpGenerator.h"

namespace JSC {

class JITRightShiftGenerator : public JITBitBinaryOpGenerator {
public:
    enum ShiftType {
        SignedShift,
        UnsignedShift
    };

    JITRightShiftGenerator(const SnippetOperand& leftOperand, const SnippetOperand& rightOperand,
        JSValueRegs result, JSValueRegs left, JSValueRegs right,
        FPRReg leftFPR, GPRReg scratchGPR, ShiftType type = SignedShift)
        : JITBitBinaryOpGenerator(leftOperand, rightOperand, result, left, right, scratchGPR)
        , m_shiftType(type)
        , m_leftFPR(leftFPR)
    { }

    void generateFastPath(CCallHelpers&);

private:
    ShiftType m_shiftType;
    FPRReg m_leftFPR;
};

} // namespace JSC

#endif // ENABLE(JIT)
