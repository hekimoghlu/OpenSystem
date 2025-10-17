/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 3, 2022.
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

#if ENABLE(WEBASSEMBLY)

#include "FPRInfo.h"
#include "GPRInfo.h"
#include "Reg.h"
#include <wtf/PrintStream.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

namespace Wasm {

class ValueLocation {
    WTF_MAKE_TZONE_ALLOCATED(ValueLocation);
public:
    enum Kind : uint8_t {
        GPRRegister,
        FPRRegister,
        Stack,
        StackArgument,
    };

    ValueLocation()
        : m_kind(GPRRegister)
    {
    }

    explicit ValueLocation(JSValueRegs regs)
        : m_kind(GPRRegister)
    {
        u.jsr = regs;
    }

    explicit ValueLocation(FPRReg reg)
        : m_kind(FPRRegister)
    {
        u.fpr = reg;
    }

    static ValueLocation stack(intptr_t offsetFromFP)
    {
        ValueLocation result;
        result.m_kind = Stack;
        result.u.offsetFromFP = offsetFromFP;
        return result;
    }

    static ValueLocation stackArgument(intptr_t offsetFromSP)
    {
        ValueLocation result;
        result.m_kind = StackArgument;
        result.u.offsetFromSP = offsetFromSP;
        return result;
    }

    Kind kind() const { return m_kind; }

    bool isGPR() const { return kind() == GPRRegister; }
    bool isFPR() const { return kind() == FPRRegister; }
    bool isStack() const { return kind() == Stack; }
    bool isStackArgument() const { return kind() == StackArgument; }

    JSValueRegs jsr() const
    {
        ASSERT(isGPR());
        return u.jsr;
    }

    FPRReg fpr() const
    {
        ASSERT(isFPR());
        return u.fpr;
    }

    intptr_t offsetFromFP() const
    {
        ASSERT(isStack());
        return u.offsetFromFP;
    }

    intptr_t offsetFromSP() const
    {
        ASSERT(isStackArgument());
        return u.offsetFromSP;
    }

    JS_EXPORT_PRIVATE void dump(PrintStream&) const;

private:
    union U {
        JSValueRegs jsr;
        FPRReg fpr;
        intptr_t offsetFromFP;
        intptr_t offsetFromSP;

        U()
        {
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
            memset(static_cast<void*>(this), 0, sizeof(*this));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
        }
    } u;
    Kind m_kind;
};

} } // namespace JSC::Wasm

namespace WTF {

void printInternal(PrintStream&, JSC::Wasm::ValueLocation::Kind);

} // namespace WTF

#endif // ENABLE(WEBASSEMBLY)
