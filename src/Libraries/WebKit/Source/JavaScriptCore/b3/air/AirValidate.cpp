/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
#include "AirValidate.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirInstInlines.h"
#include "B3Procedure.h"
#include <wtf/StringPrintStream.h>

namespace JSC { namespace B3 { namespace Air {

namespace {

class Validater {
public:
    Validater(Code& code, const char* dumpBefore)
        : m_code(code)
        , m_dumpBefore(dumpBefore)
    {
    }

#define VALIDATE(condition, message) do {                               \
        if (condition)                                                  \
            break;                                                      \
        fail(__FILE__, __LINE__, WTF_PRETTY_FUNCTION, #condition, toCString message); \
    } while (false)
    
    void run()
    {
        UncheckedKeyHashSet<StackSlot*> validSlots;
        UncheckedKeyHashSet<BasicBlock*> validBlocks;
        UncheckedKeyHashSet<Special*> validSpecials;
        
        for (BasicBlock* block : m_code)
            validBlocks.add(block);
        for (StackSlot* slot : m_code.stackSlots())
            validSlots.add(slot);
        for (Special* special : m_code.specials())
            validSpecials.add(special);

        for (BasicBlock* block : m_code) {
            // Blocks that are entrypoints must not have predecessors.
            if (m_code.isEntrypoint(block))
                VALIDATE(!block->numPredecessors(), ("At entrypoint ", *block));
            
            for (unsigned instIndex = 0; instIndex < block->size(); ++instIndex) {
                Inst& inst = block->at(instIndex);
                for (Arg& arg : inst.args) {
                    switch (arg.kind()) {
                    case Arg::Stack:
                        VALIDATE(validSlots.contains(arg.stackSlot()), ("At ", inst, " in ", *block));
                        break;
                    case Arg::Special:
                        VALIDATE(validSpecials.contains(arg.special()), ("At ", inst, " in ", *block));
                        break;
                    default:
                        break;
                    }
                }
                VALIDATE(inst.isValidForm(), ("At ", inst, " in ", *block));
                if (instIndex == block->size() - 1)
                    VALIDATE(inst.isTerminal(), ("At ", inst, " in ", *block));
                else
                    VALIDATE(!inst.isTerminal(), ("At ", inst, " in ", *block));

                // forEachArg must return Arg&'s that point into the args array.
                inst.forEachArg(
                    [&] (Arg& arg, Arg::Role role, Bank, Width width) {
                        VALIDATE(&arg >= &inst.args[0], ("At ", arg, " in ", inst, " in ", *block));
                        VALIDATE(&arg <= &inst.args.last(), ("At ", arg, " in ", inst, " in ", *block));

                        // FIXME: replace with a check for wasm simd instructions.
                        VALIDATE(Options::useWasmSIMD()
                            || !Arg::isAnyUse(role)
                            || width <= Width64, ("At ", inst, " in ", *block, " arg ", arg));
                    });
                
                switch (inst.kind.opcode) {
                case EntrySwitch:
                    VALIDATE(block->numSuccessors() == m_code.proc().numEntrypoints(), ("At ", inst, " in ", *block));
                    break;
                case Shuffle:
                    // We can't handle trapping shuffles because of how we lower them. That could
                    // be fixed though. Ditto for shuffles that would do fences, which is the other
                    // use of this bit.
                    VALIDATE(!inst.kind.effects, ("At ", inst, " in ", *block));
                    break;
                case VectorExtendLow:
                case VectorExtendHigh:
                    VALIDATE(elementByteSize(inst.args[0].simdInfo().lane) <= 8, ("At ", inst, " in ", *block));
                    VALIDATE(elementByteSize(inst.args[0].simdInfo().lane) >= 2, ("At ", inst, " in ", *block));
                    break;
                case ExtractRegister64:
                    VALIDATE(inst.args[2].isImm(), ("At ", inst, " in ", *block));
                    VALIDATE(inst.args[2].asTrustedImm32().m_value < 64, ("At ", inst, " in ", *block));
                    break;
                case ExtractRegister32:
                    VALIDATE(inst.args[2].isImm(), ("At ", inst, " in ", *block));
                    VALIDATE(inst.args[2].asTrustedImm32().m_value < 32, ("At ", inst, " in ", *block));
                    break;
                default:
                    break;
                }
            }
            for (BasicBlock* successor : block->successorBlocks())
                VALIDATE(validBlocks.contains(successor), ("In ", *block));
        }

        for (BasicBlock* block : m_code) {
            // We expect the predecessor list to be de-duplicated.
            UncheckedKeyHashSet<BasicBlock*> predecessors;
            for (BasicBlock* predecessor : block->predecessors())
                predecessors.add(predecessor);
            VALIDATE(block->numPredecessors() == predecessors.size(), ("At ", *block));
        }
    }

private:
    NO_RETURN_DUE_TO_CRASH void fail(
        const char* filename, int lineNumber, const char* function, const char* condition,
        CString message)
    {
        CString failureMessage;
        {
            StringPrintStream out;
            out.print("AIR VALIDATION FAILURE\n");
            out.print("    ", condition, " (", filename, ":", lineNumber, ")\n");
            out.print("    ", message, "\n");
            out.print("    After ", m_code.lastPhaseName(), "\n");
            failureMessage = out.toCString();
        }

        dataLog(failureMessage);
        if (m_dumpBefore) {
            dataLog("Before ", m_code.lastPhaseName(), ":\n");
            dataLog(m_dumpBefore);
        }
        dataLog("At time of failure:\n");
        dataLog(m_code);

        dataLog(failureMessage);
        WTFReportAssertionFailure(filename, lineNumber, function, condition);
        CRASH();
    }
    
    Code& m_code;
    const char* m_dumpBefore;
};

} // anonymous namespace

void validate(Code& code, const char* dumpBefore)
{
    Validater validater(code, dumpBefore);
    validater.run();
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

