/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include "AirLowerStackArgs.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirInsertionSet.h"
#include "AirInstInlines.h"
#include "AirPhaseScope.h"

namespace JSC { namespace B3 { namespace Air {

void lowerStackArgs(Code& code)
{
    PhaseScope phaseScope(code, "lowerStackArgs"_s);
    
    // Now we need to deduce how much argument area we need.
    for (BasicBlock* block : code) {
        for (Inst& inst : *block) {
            for (Arg& arg : inst.args) {
                if (arg.isCallArg()) {
                    ASSERT(arg.offset() >= 0);
                    // We always check the conservative register bytes for Bank::FP because
                    // CallArgs do not store which bank they are.
                    code.requestCallArgAreaSizeInBytes(arg.offset() + (code.usesSIMD() ? conservativeRegisterBytes(Bank::FP) : conservativeRegisterBytesWithoutVectors(Bank::FP)));
                }
            }
        }
    }

    code.setFrameSize(code.frameSize() + code.callArgAreaSizeInBytes());

    // Finally, transform the code to use Addr's instead of StackSlot's. This is a lossless
    // transformation since we can search the StackSlots array to figure out which StackSlot any
    // offset-from-FP refers to.

    InsertionSet insertionSet(code);
    for (BasicBlock* block : code) {
        // FIXME We can keep track of the last large offset which was materialized in this block, and reuse the register
        // if it hasn't been clobbered instead of renetating imm+add+addr every time. https://bugs.webkit.org/show_bug.cgi?id=171387

        for (unsigned instIndex = 0; instIndex < block->size(); ++instIndex) {
            Inst& inst = block->at(instIndex);

            if (isARM64() && (inst.kind.opcode == Lea32 || inst.kind.opcode == Lea64)) {
                // On ARM64, Lea is just an add. We can't handle this below because
                // taking into account the Width to see if we can compute the immediate
                // is wrong.
                auto lowerArmLea = [&] (Value::OffsetType offset, Tmp base) {
                    ASSERT(inst.args[1].isTmp());

                    if (Arg::isValidImmForm(offset))
                        inst = Inst(inst.kind.opcode == Lea32 ? Add32 : Add64, inst.origin, Arg::imm(offset), base, inst.args[1]);
                    else {
                        Air::Tmp tmp = Air::Tmp(extendedOffsetAddrRegister());
                        Arg offsetArg = Arg::bigImm(offset);
                        insertionSet.insert(instIndex, Move, inst.origin, offsetArg, tmp);
                        inst = Inst(inst.kind.opcode == Lea32 ? Add32 : Add64, inst.origin, tmp, base, inst.args[1]);
                    }
                };

                switch (inst.args[0].kind()) {
                case Arg::Stack: {
                    StackSlot* slot = inst.args[0].stackSlot();
                    lowerArmLea(inst.args[0].offset() + slot->offsetFromFP(), Tmp(GPRInfo::callFrameRegister));
                    break;
                }
                case Arg::CallArg:
                    lowerArmLea(inst.args[0].offset() - code.frameSize(), Tmp(GPRInfo::callFrameRegister));
                    break;
                case Arg::Addr:
                    lowerArmLea(inst.args[0].offset(), inst.args[0].base());
                    break;
                case Arg::ExtendedOffsetAddr:
                    ASSERT_NOT_REACHED();
                    break;
                default:
                    break;
                }

                continue;
            }

            inst.forEachArg(
                [&] (Arg& arg, Arg::Role role, Bank, Width width) {
                    auto stackAddr = [&] (unsigned instIndex, Value::OffsetType offsetFromFP) -> Arg {
                        int32_t offsetFromSP = offsetFromFP + code.frameSize();

                        if (inst.admitsExtendedOffsetAddr(arg)) {
                            // Stackmaps and patchpoints expect addr inputs relative to SP or FP only. We might as well
                            // not even bother generating an addr with valid form for these opcodes since extended offset
                            // addr is always valid.
                            return Arg::extendedOffsetAddr(offsetFromFP);
                        }

                        Arg result = Arg::addr(Air::Tmp(GPRInfo::callFrameRegister), offsetFromFP);
                        if (result.isValidForm(Move, width))
                            return result;

                        result = Arg::addr(Air::Tmp(MacroAssembler::stackPointerRegister), offsetFromSP);
                        if (result.isValidForm(Move, width))
                            return result;

                        if (inst.kind.opcode == Patch)
                            return Arg::extendedOffsetAddr(offsetFromFP);

#if CPU(ARM64) || CPU(RISCV64)
                        Air::Tmp tmp = Air::Tmp(extendedOffsetAddrRegister());

                        Arg largeOffset = Arg::isValidImmForm(offsetFromSP) ? Arg::imm(offsetFromSP) : Arg::bigImm(offsetFromSP);
                        insertionSet.insert(instIndex, Move, inst.origin, largeOffset, tmp);
                        insertionSet.insert(instIndex, Add64, inst.origin, Air::Tmp(MacroAssembler::stackPointerRegister), tmp);
                        result = Arg::addr(tmp, 0);
                        return result;
#elif CPU(ARM)
                        // We solve this in AirAllocateRegistersAndStackAndGenerateCode.cpp.
                        UNUSED_PARAM(instIndex);
                        return result;
#elif CPU(X86_64)
                        UNUSED_PARAM(instIndex);
                        // Can't happen on x86: immediates are always big enough for frame size.
                        RELEASE_ASSERT_NOT_REACHED();
#else
#error Unhandled architecture.
#endif
                    };
                    
                    switch (arg.kind()) {
                    case Arg::Stack: {
                        StackSlot* slot = arg.stackSlot();
                        if (inst.kind.opcode == Move && slot->kind() == StackSlotKind::Spill)
                            inst.kind.spill = true;
                        if (Arg::isZDef(role)
                            && slot->kind() == StackSlotKind::Spill
                            && slot->byteSize() > bytesForWidth(width)) {
                            // Currently we only handle this simple case because it's the only one
                            // that arises: ZDef's are only 32-bit right now. So, when we hit these
                            // assertions it means that we need to implement those other kinds of
                            // zero fills.
                            RELEASE_ASSERT(slot->byteSize() == 8);
                            RELEASE_ASSERT(width == Width32);

#if CPU(ARM64) || CPU(RISCV64)
                            Air::Opcode storeOpcode = Move32;
                            Air::Arg::Kind operandKind = Arg::ZeroReg;
                            Air::Arg operand = Arg::zeroReg();
#elif CPU(X86_64) || CPU(ARM)
                            Air::Opcode storeOpcode = Move32;
                            Air::Arg::Kind operandKind = Arg::Imm;
                            Air::Arg operand = Arg::imm(0);
#else
#error Unhandled architecture.
#endif
                            RELEASE_ASSERT(isValidForm(storeOpcode, operandKind, Arg::Stack));
                            insertionSet.insert(
                                instIndex + 1, storeOpcode, inst.origin, operand,
                                stackAddr(instIndex + 1, arg.offset() + 4 + slot->offsetFromFP()));
                        }
                        arg = stackAddr(instIndex, arg.offset() + slot->offsetFromFP());
                        break;
                    }
                    case Arg::CallArg:
                        arg = stackAddr(instIndex, arg.offset() - code.frameSize());
                        break;
                    default:
                        break;
                    }
                }
            );
        }
        insertionSet.execute(block);
    }
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

