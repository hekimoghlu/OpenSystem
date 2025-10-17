/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#include "AirEliminateDeadCode.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirInstInlines.h"
#include "AirPhaseScope.h"
#include "AirTmpInlines.h"
#include "AirTmpSet.h"
#include <wtf/IndexSet.h>

namespace JSC { namespace B3 { namespace Air {

bool eliminateDeadCode(Code& code)
{
    PhaseScope phaseScope(code, "eliminateDeadCode"_s);

    TmpSet liveTmps;
    IndexSet<StackSlot*> liveStackSlots;
    bool changed { false };
    
    auto isArgLive = [&] (const Arg& arg) -> bool {
        switch (arg.kind()) {
        case Arg::Tmp:
            if (arg.isReg())
                return true;
            return liveTmps.contains(arg.tmp());
        case Arg::Stack:
            if (arg.stackSlot()->isLocked())
                return true;
            return liveStackSlots.contains(arg.stackSlot());
        default:
            return true;
        }
    };

    auto isInstLive = [&] (Inst& inst) -> bool {
        if (inst.hasNonArgEffects())
            return true;

        // This instruction should be presumed dead, if its Args are all dead.
        bool storesToLive = false;
        inst.forEachArg(
            [&] (Arg& arg, Arg::Role role, Bank, Width) {
                if (!Arg::isAnyDef(role))
                    return;
                if (role == Arg::Scratch)
                    return;
                storesToLive |= isArgLive(arg);
            });
        return storesToLive;
    };
    
    // Returns true if it's live.
    auto handleInst = [&] (Inst& inst) -> bool {
        if (!isInstLive(inst))
            return false;

        // We get here if the Inst is live. For simplicity we say that a live instruction forces
        // liveness upon everything it mentions.
        for (Arg& arg : inst.args) {
            if (arg.isStack() && !arg.stackSlot()->isLocked())
                changed |= liveStackSlots.add(arg.stackSlot());
            arg.forEachTmpFast(
                [&] (Tmp& tmp) {
                    if (!tmp.isReg())
                        changed |= liveTmps.add(tmp);
                });
        }
        return true;
    };

    Vector<Inst*> possiblyDead;
    
    for (BasicBlock* block : code) {
        for (Inst& inst : *block) {
            if (!handleInst(inst))
                possiblyDead.append(&inst);
        }
    }
    
    auto runForward = [&] () -> bool {
        changed = false;
        possiblyDead.removeAllMatching(
            [&] (Inst* inst) -> bool {
                bool result = handleInst(*inst);
                changed |= result;
                return result;
            });
        return changed;
    };

    auto runBackward = [&] () -> bool {
        changed = false;
        for (unsigned i = possiblyDead.size(); i--;) {
            bool result = handleInst(*possiblyDead[i]);
            if (result) {
                possiblyDead[i] = possiblyDead.last();
                possiblyDead.removeLast();
                changed = true;
            }
        }
        return changed;
    };

    for (;;) {
        // Propagating backward is most likely to be profitable.
        if (!runBackward())
            break;
        if (!runBackward())
            break;

        // Occasionally propagating forward greatly reduces the likelihood of pathologies.
        if (!runForward())
            break;
    }

    unsigned removedInstCount = 0;
    for (BasicBlock* block : code) {
        removedInstCount += block->insts().removeAllMatching(
            [&] (Inst& inst) -> bool {
                return !isInstLive(inst);
            });
    }

    return !!removedInstCount;
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

