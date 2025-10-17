/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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

// This is guarded by ENABLE_DFG_JIT only because it uses some value profiles
// that are currently only used if the DFG is enabled (i.e. they are not
// available in the profile-only configuration). Hopefully someday all of
// these #if's will disappear...
#if ENABLE(DFG_JIT)

#include "BytecodeIndex.h"
#include "CodeOrigin.h"
#include "GPRInfo.h"
#include "Operands.h"
#include "TagRegistersMode.h"

namespace JSC {

class UnaryArithProfile;
class BinaryArithProfile;
class CCallHelpers;
class CodeBlock;
struct ValueProfile;

class MethodOfGettingAValueProfile {
public:
    MethodOfGettingAValueProfile()
        : m_kind(Kind::None)
    {
    }

    static MethodOfGettingAValueProfile unaryArithProfile(CodeOrigin codeOrigin)
    {
        MethodOfGettingAValueProfile result;
        result.m_kind = Kind::UnaryArithProfile;
        result.m_codeOrigin = codeOrigin;
        return result;
    }

    static MethodOfGettingAValueProfile binaryArithProfile(CodeOrigin codeOrigin)
    {
        MethodOfGettingAValueProfile result;
        result.m_kind = Kind::BinaryArithProfile;
        result.m_codeOrigin = codeOrigin;
        return result;
    }

    static MethodOfGettingAValueProfile argumentValueProfile(CodeOrigin codeOrigin, Operand operand)
    {
        MethodOfGettingAValueProfile result;
        result.m_kind = Kind::ArgumentValueProfile;
        result.m_codeOrigin = codeOrigin;
        result.m_rawOperand = operand.asBits();
        return result;
    }

    static MethodOfGettingAValueProfile bytecodeValueProfile(CodeOrigin codeOrigin)
    {
        MethodOfGettingAValueProfile result;
        result.m_kind = Kind::BytecodeValueProfile;
        result.m_codeOrigin = codeOrigin;
        return result;
    }

    static MethodOfGettingAValueProfile lazyOperandValueProfile(CodeOrigin codeOrigin, Operand operand)
    {
        MethodOfGettingAValueProfile result;
        result.m_kind = Kind::LazyOperandValueProfile;
        result.m_codeOrigin = codeOrigin;
        result.m_rawOperand = operand.asBits();
        return result;
    }

    explicit operator bool() const { return m_kind != Kind::None; }

    // The temporary register is only needed on 64-bits builds (for testing BigInt32).
    void emitReportValue(CCallHelpers&, CodeBlock* optimizedCodeBlock, JSValueRegs, GPRReg tempGPR, TagRegistersMode = HaveTagRegisters) const;

private:
    enum class Kind : uint8_t {
        None,
        UnaryArithProfile,
        BinaryArithProfile,
        BytecodeValueProfile,
        ArgumentValueProfile,
        LazyOperandValueProfile,
    };
    static constexpr unsigned bitsOfKind = 3;
    static_assert(static_cast<unsigned>(Kind::LazyOperandValueProfile) <= ((1U << bitsOfKind) - 1));

    CodeOrigin m_codeOrigin;
    uint64_t m_rawOperand : Operand::maxBits { 0 };
    Kind m_kind : bitsOfKind;
};

} // namespace JSC

#endif // ENABLE(DFG_JIT)
