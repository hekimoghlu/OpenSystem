/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 26, 2023.
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
#include "CallFrameShuffleData.h"

#if ENABLE(JIT)

#include "BaselineJITRegisters.h"
#include "BytecodeStructs.h"
#include "CodeBlock.h"
#include "RegisterAtOffsetList.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CallFrameShuffleData);

void CallFrameShuffleData::setupCalleeSaveRegisters(const RegisterAtOffsetList* registerSaveLocations)
{
    auto calleeSaveRegisters = RegisterSetBuilder::vmCalleeSaveRegisters();

    for (size_t i = 0; i < registerSaveLocations->registerCount(); ++i) {
        RegisterAtOffset entry { registerSaveLocations->at(i) };
        if (!calleeSaveRegisters.contains(entry.reg(), IgnoreVectors))
            continue;

        int saveSlotIndexInCPURegisters = entry.offsetAsIndex();

#if USE(JSVALUE64)
        // CPU registers are the same size as virtual registers
        VirtualRegister saveSlot { saveSlotIndexInCPURegisters };
        registers[entry.reg()]
            = ValueRecovery::displacedInJSStack(saveSlot, DataFormatJS);
#elif USE(JSVALUE32_64)
        // On 32-bit architectures, 2 callee saved GPRs may be packed into the same slot
        if (entry.reg().isGPR()) {
            static_assert(!PayloadOffset || !TagOffset);
            static_assert(PayloadOffset == 4 || TagOffset == 4);
            bool inTag = (saveSlotIndexInCPURegisters & 1) == !!TagOffset;
            if (saveSlotIndexInCPURegisters < 0)
                saveSlotIndexInCPURegisters -= 1; // Round towards -inf
            VirtualRegister saveSlot { saveSlotIndexInCPURegisters / 2 };
            registers[entry.reg()] = ValueRecovery::calleeSaveGPRDisplacedInJSStack(saveSlot, inTag);
        } else {
            ASSERT(!(saveSlotIndexInCPURegisters & 1)); // Should be at an even offset
            VirtualRegister saveSlot { saveSlotIndexInCPURegisters / 2 };
            registers[entry.reg()] = ValueRecovery::displacedInJSStack(saveSlot, DataFormatDouble);
        }
#endif
    }

    for (Reg reg = Reg::first(); reg <= Reg::last(); reg = reg.next()) {
        if (!calleeSaveRegisters.contains(reg, IgnoreVectors))
            continue;

        if (registers[reg])
            continue;

#if USE(JSVALUE64)
        registers[reg] = ValueRecovery::inRegister(reg, DataFormatJS);
#elif USE(JSVALUE32_64)
        registers[reg] = ValueRecovery::inRegister(reg, reg.isGPR() ? DataFormatInt32 : DataFormatDouble);
#endif
    }
}

CallFrameShuffleData CallFrameShuffleData::createForBaselineOrLLIntTailCall(const OpTailCall& bytecode, unsigned numParameters)
{
    CallFrameShuffleData shuffleData;
    shuffleData.numPassedArgs = bytecode.m_argc;
    shuffleData.numParameters = numParameters;
#if USE(JSVALUE64)
    shuffleData.numberTagRegister = GPRInfo::numberTagRegister;
#endif
    shuffleData.numLocals = bytecode.m_argv - sizeof(CallerFrameAndPC) / sizeof(Register);
    shuffleData.args.grow(bytecode.m_argc);
    for (unsigned i = 0; i < bytecode.m_argc; ++i) {
        shuffleData.args[i] =
            ValueRecovery::displacedInJSStack(
                virtualRegisterForArgumentIncludingThis(i) - bytecode.m_argv,
                DataFormatJS);
    }
#if USE(JSVALUE64)
    shuffleData.callee = ValueRecovery::inGPR(BaselineJITRegisters::Call::calleeJSR.payloadGPR(), DataFormatJS);
#elif USE(JSVALUE32_64)
    shuffleData.callee = ValueRecovery::inPair(BaselineJITRegisters::Call::calleeJSR.tagGPR(), BaselineJITRegisters::Call::calleeJSR.payloadGPR());
#endif
    shuffleData.setupCalleeSaveRegisters(&RegisterAtOffsetList::llintBaselineCalleeSaveRegisters());
    shuffleData.shrinkToFit();
    return shuffleData;
}

} // namespace JSC

#endif // ENABLE(JIT)
