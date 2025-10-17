/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
#include "AirReportUsedRegisters.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirInstInlines.h"
#include "AirPadInterference.h"
#include "AirRegLiveness.h"
#include "AirPhaseScope.h"

namespace JSC { namespace B3 { namespace Air {

void reportUsedRegisters(Code& code)
{
    PhaseScope phaseScope(code, "reportUsedRegisters"_s);
    
    static constexpr bool verbose = false;

    padInterference(code);
    
    if (verbose)
        dataLog("Doing reportUsedRegisters on:\n", code);

    RegLiveness liveness(code);

    for (BasicBlock* block : code) {
        if (verbose)
            dataLog("Looking at: ", *block, "\n");
        
        RegLiveness::LocalCalc localCalc(liveness, block);

        for (unsigned instIndex = block->size(); instIndex--;) {
            Inst& inst = block->at(instIndex);
            
            if (verbose)
                dataLog("   Looking at: ", inst, "\n");

            // Kill dead assignments to registers. For simplicity we say that a store is killable if
            // it has only late defs and those late defs are to registers that are dead right now.
            if (!inst.hasNonArgEffects()) {
                bool canDelete = true;
                inst.forEachArg(
                    [&] (Arg& arg, Arg::Role role, Bank, Width) {
                        if (Arg::isEarlyDef(role)) {
                            if (verbose)
                                dataLog("        Cannot delete because of ", arg, "\n");
                            canDelete = false;
                            return;
                        }
                        if (!Arg::isLateDef(role))
                            return;
                        if (!arg.isReg()) {
                            if (verbose)
                                dataLog("        Cannot delete because arg is not reg: ", arg, "\n");
                            canDelete = false;
                            return;
                        }
                        if (localCalc.isLive(arg.reg())) {
                            if (verbose)
                                dataLog("        Cannot delete because arg is live: ", arg, "\n");
                            canDelete = false;
                            return;
                        }
                    });
                if (canDelete)
                    inst = Inst();
            }

            if (inst.kind.opcode == Patch)
                inst.reportUsedRegisters(localCalc.live());
            localCalc.execute(instIndex);
        }
        
        block->insts().removeAllMatching(
            [&] (const Inst& inst) -> bool {
                return !inst;
            });
    }

    if (verbose)
        dataLog("After reportUsedRegisters:\n", code);
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)


