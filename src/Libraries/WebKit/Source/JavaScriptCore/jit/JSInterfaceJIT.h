/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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

#include "BytecodeConventions.h"
#include "CCallHelpers.h"
#include "FPRInfo.h"
#include "GPRInfo.h"
#include "JSCJSValue.h"
#include "JSString.h"
#include "MacroAssembler.h"
#include <wtf/TZoneMalloc.h>

#if ENABLE(JIT)

namespace JSC {
    class JSInterfaceJIT : public CCallHelpers, public GPRInfo, public JSRInfo, public FPRInfo {
        WTF_MAKE_TZONE_NON_HEAP_ALLOCATABLE(JSInterfaceJIT);
    public:

        JSInterfaceJIT(VM* vm = nullptr, CodeBlock* codeBlock = nullptr)
            : CCallHelpers(codeBlock)
            , m_vm(vm)
        {
        }

        inline Jump emitLoadJSCell(VirtualRegister, RegisterID payload);
        inline Jump emitLoadInt32(VirtualRegister, RegisterID dst);
        inline Jump emitLoadDouble(VirtualRegister, FPRegisterID dst, RegisterID scratch);

        inline void emitLoadJSValue(VirtualRegister, JSValueRegs dst);

        VM* vm() const { return m_vm; }

        VM* const m_vm;
    };

#if USE(JSVALUE32_64)
    inline JSInterfaceJIT::Jump JSInterfaceJIT::emitLoadJSCell(VirtualRegister virtualRegister, RegisterID payload)
    {
        ASSERT(virtualRegister < VirtualRegister(FirstConstantRegisterIndex));
        loadPtr(payloadFor(virtualRegister), payload);
        return branchIfNotCell(tagFor(virtualRegister));
    }

    inline JSInterfaceJIT::Jump JSInterfaceJIT::emitLoadInt32(VirtualRegister virtualRegister, RegisterID dst)
    {
        ASSERT(virtualRegister < VirtualRegister(FirstConstantRegisterIndex));
        loadPtr(payloadFor(virtualRegister), dst);
        return branch32(NotEqual, tagFor(virtualRegister), TrustedImm32(JSValue::Int32Tag));
    }

    inline JSInterfaceJIT::Jump JSInterfaceJIT::emitLoadDouble(VirtualRegister virtualRegister, FPRegisterID dst, RegisterID scratch)
    {
        ASSERT(virtualRegister < VirtualRegister(FirstConstantRegisterIndex));
        loadPtr(tagFor(virtualRegister), scratch);
        Jump isDouble = branch32(Below, scratch, TrustedImm32(JSValue::LowestTag));
        Jump notInt = branch32(NotEqual, scratch, TrustedImm32(JSValue::Int32Tag));
        loadPtr(payloadFor(virtualRegister), scratch);
        convertInt32ToDouble(scratch, dst);
        Jump done = jump();
        isDouble.link(this);
        loadDouble(addressFor(virtualRegister), dst);
        done.link(this);
        return notInt;
    }
#elif USE(JSVALUE64)
    inline JSInterfaceJIT::Jump JSInterfaceJIT::emitLoadJSCell(VirtualRegister virtualRegister, RegisterID dst)
    {
        load64(addressFor(virtualRegister), dst);
        return branchIfNotCell(dst);
    }

    inline JSInterfaceJIT::Jump JSInterfaceJIT::emitLoadInt32(VirtualRegister virtualRegister, RegisterID dst)
    {
        load64(addressFor(virtualRegister), dst);
        Jump notInt32 = branchIfNotInt32(dst);
        zeroExtend32ToWord(dst, dst);
        return notInt32;
    }

    inline JSInterfaceJIT::Jump JSInterfaceJIT::emitLoadDouble(VirtualRegister virtualRegister, FPRegisterID dst, RegisterID scratch)
    {
        load64(addressFor(virtualRegister), scratch);
        Jump notNumber = branchIfNotNumber(scratch);
        Jump notInt = branchIfNotInt32(scratch);
        convertInt32ToDouble(scratch, dst);
        Jump done = jump();
        notInt.link(this);
        unboxDouble(scratch, scratch, dst);
        done.link(this);
        return notNumber;
    }
#endif

inline void JSInterfaceJIT::emitLoadJSValue(VirtualRegister virtualRegister, JSValueRegs dst)
{
    loadValue(addressFor(virtualRegister), dst);
}

} // namespace JSC

#endif // ENABLE(JIT)
