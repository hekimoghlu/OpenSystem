/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
#include "AirHandleCalleeSaves.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirInstInlines.h"
#include "RegisterSet.h"

namespace JSC { namespace B3 { namespace Air {

void handleCalleeSaves(Code& code)
{
    RegisterSetBuilder usedCalleeSaves;

    for (BasicBlock* block : code) {
        for (Inst& inst : *block) {
            inst.forEachTmpFast(
                [&] (Tmp& tmp) {
                    // At first we just record all used regs.
                    usedCalleeSaves.add(tmp.reg(), IgnoreVectors);
                });

            if (inst.kind.opcode == Patch) {
                usedCalleeSaves.merge(inst.extraClobberedRegs());
                usedCalleeSaves.merge(inst.extraEarlyClobberedRegs());
            }
        }
    }

    handleCalleeSaves(code, WTFMove(usedCalleeSaves));
}

void handleCalleeSaves(Code& code, RegisterSetBuilder usedCalleeSaves)
{
    // We filter to really get the callee saves.
    usedCalleeSaves.filter(RegisterSetBuilder::calleeSaveRegisters());
    usedCalleeSaves.filter(code.mutableRegs());
    usedCalleeSaves.exclude(RegisterSetBuilder::stackRegisters()); // We don't need to save FP here.

#if CPU(ARM)
    // See AirCode for a similar comment about why ARMv7 acts weird here.
    // Essentially, we might at any point clobber this, and it is a callee-save.
    // This should be fixed.
    usedCalleeSaves.add(MacroAssembler::addressTempRegister, IgnoreVectors);
#endif

    auto calleSavesToSave = usedCalleeSaves.buildAndValidate();

    if (!calleSavesToSave.numberOfSetRegisters())
        return;

    RegisterAtOffsetList calleeSaveRegisters = RegisterAtOffsetList(calleSavesToSave);

    size_t byteSize = 0;
    for (const RegisterAtOffset& entry : calleeSaveRegisters)
        byteSize = std::max(static_cast<size_t>(-entry.offset()), byteSize);
    ASSERT(calleeSaveRegisters.sizeOfAreaInBytes() == byteSize);

    code.setCalleeSaveRegisterAtOffsetList(
        WTFMove(calleeSaveRegisters),
        code.addStackSlot(byteSize, StackSlotKind::Locked));
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
