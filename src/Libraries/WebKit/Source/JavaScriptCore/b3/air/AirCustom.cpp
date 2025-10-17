/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include "AirCustom.h"

#if ENABLE(B3_JIT)

#include "AirCCallingConvention.h"
#include "AirInstInlines.h"
#include "B3CCallValue.h"
#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"
#include "CCallHelpers.h"

namespace JSC { namespace B3 { namespace Air {

bool PatchCustom::isValidForm(Inst& inst)
{
    if (inst.args.size() < 1)
        return false;
    if (!inst.args[0].isSpecial())
        return false;
    if (!inst.args[0].special()->isValid(inst))
        return false;
    auto clobberedEarly = inst.extraEarlyClobberedRegs().filter(RegisterSetBuilder::allScalarRegisters()).buildAndValidate();
    auto clobberedLate = inst.extraClobberedRegs().filter(RegisterSetBuilder::allScalarRegisters()).buildAndValidate();
    bool ok = true;
    inst.forEachTmp(
        [&] (Tmp& tmp, Arg::Role role, Bank, Width) {
            if (!tmp.isReg())
                return;
            if (Arg::isLateDef(role) || Arg::isLateUse(role))
                ok &= !clobberedLate.contains(tmp.reg(), IgnoreVectors);
            else
                ok &= !clobberedEarly.contains(tmp.reg(), IgnoreVectors);
        });
    return ok;
}

bool CCallCustom::isValidForm(Inst& inst)
{
    CCallValue* value = inst.origin->as<CCallValue>();
    if (!value)
        return false;

    if (!inst.args[0].isSpecial())
        return false;

    Special* special = inst.args[0].special();
    Code& code = special->code();

    size_t resultCount = cCallResultCount(code, value);
    size_t expectedArgCount = resultCount + 1; // first Arg is always CCallSpecial.
    for (Value* child : value->children()) {
        ASSERT(!child->type().isTuple());
        expectedArgCount += cCallArgumentRegisterCount(child->type());
    }

    if (inst.args.size() != expectedArgCount)
        return false;

    // The arguments can only refer to the stack, tmps, or immediates.
    for (unsigned i = inst.args.size() - 1; i; --i) {
        Arg arg = inst.args[i];
        if (!arg.isTmp() && !arg.isStackMemory() && !arg.isSomeImm())
            return false;
    }

    // Callee
    if (!inst.args[1].isGP())
        return false;

    unsigned offset = 2;

    // If there is a result then it cannot be an immediate.
    for (size_t i = 0; i < resultCount; ++i) {
        if (inst.args[offset].isSomeImm())
            return false;

        if (value->type().isTuple()) {
            Type type = code.proc().typeAtOffset(value->type(), i);
            if (!inst.args[offset].canRepresent(type))
                return false;
        } else if (!inst.args[offset].canRepresent(value))
            return false;
        offset++;
    }

    auto checkNextArg = [&](Value* child) {
        return inst.args[offset++].canRepresent(child);
    };

    for (unsigned i = 1 ; i < value->numChildren(); ++i) {
        Value* child = value->child(i);
        for (unsigned j = 0; j < cCallArgumentRegisterCount(child->type()); j++) {
            if (!checkNextArg(child))
                return false;
        }
    }

    return true;
}

MacroAssembler::Jump CCallCustom::generate(Inst& inst, CCallHelpers&, GenerationContext&)
{
    dataLog("FATAL: Unlowered C call: ", inst, "\n");
    UNREACHABLE_FOR_PLATFORM();
    return MacroAssembler::Jump();
}

bool ShuffleCustom::isValidForm(Inst& inst)
{
    if (inst.args.size() % 3)
        return false;

    // A destination may only appear once. This requirement allows us to avoid the undefined behavior
    // of having a destination that is supposed to get multiple inputs simultaneously. It also
    // imposes some interesting constraints on the "shape" of the shuffle. If we treat a shuffle pair
    // as an edge and the Args as nodes, then the single-destination requirement means that the
    // shuffle graph consists of two kinds of subgraphs:
    //
    // - Spanning trees. We call these shifts. They can be executed as a sequence of Move
    //   instructions and don't usually require scratch registers.
    //
    // - Closed loops. These loops consist of nodes that have one successor and one predecessor, so
    //   there is no way to "get into" the loop from outside of it. These can be executed using swaps
    //   or by saving one of the Args to a scratch register and executing it as a shift.
    UncheckedKeyHashSet<Arg> dsts;

    for (unsigned i = 0; i < inst.args.size(); ++i) {
        Arg arg = inst.args[i];
        unsigned mode = i % 3;

        if (mode == 2) {
            // It's the width.
            if (!arg.isWidthArg())
                return false;
            continue;
        }

        // The source can be an immediate.
        if (!mode) {
            if (arg.isSomeImm())
                continue;

            if (!arg.isCompatibleBank(inst.args[i + 1]))
                return false;
        } else {
            ASSERT(mode == 1);
            if (!dsts.add(arg).isNewEntry)
                return false;
        }

        if (arg.isTmp() || arg.isMemory())
            continue;

        return false;
    }

    // No destination register may appear in any address expressions. The lowering can't handle it
    // and it's not useful for the way we end up using Shuffles. Normally, Shuffles only used for
    // stack addresses and non-stack registers.
    for (Arg& arg : inst.args) {
        if (!arg.isMemory())
            continue;
        bool ok = true;
        arg.forEachTmpFast(
            [&] (Tmp tmp) {
                if (dsts.contains(tmp))
                    ok = false;
            });
        if (!ok)
            return false;
    }

    return true;
}

MacroAssembler::Jump ShuffleCustom::generate(Inst& inst, CCallHelpers&, GenerationContext&)
{
    dataLog("FATAL: Unlowered shuffle: ", inst, "\n");
    UNREACHABLE_FOR_PLATFORM();
    return MacroAssembler::Jump();
}

bool WasmBoundsCheckCustom::isValidForm(Inst& inst)
{
    if (inst.args.size() != 2)
        return false;
    if (!inst.args[0].isTmp() && !inst.args[0].isSomeImm())
        return false;

    return inst.args[1].isReg() || inst.args[1].isTmp() || inst.args[1].isSomeImm();
}

MacroAssembler::Jump WasmBoundsCheckCustom::generate(Inst& inst, CCallHelpers& jit, GenerationContext& context)
{
    WasmBoundsCheckValue* value = inst.origin->as<WasmBoundsCheckValue>();
    MacroAssembler::Jump outOfBounds = Inst((is32Bit() ? Air::Branch32 : Air::Branch64), value, Arg::relCond(MacroAssembler::AboveOrEqual), inst.args[0], inst.args[1]).generate(jit, context);

    context.latePaths.append(createSharedTask<GenerationContext::LatePathFunction>(
        [outOfBounds, value] (CCallHelpers& jit, Air::GenerationContext& context) {
            outOfBounds.link(&jit);
            switch (value->boundsType()) {
            case WasmBoundsCheckValue::Type::Pinned:
                context.code->wasmBoundsCheckGenerator()->run(jit, value, value->bounds().pinnedSize);
                break;

            case WasmBoundsCheckValue::Type::Maximum:
                context.code->wasmBoundsCheckGenerator()->run(jit, value, InvalidGPRReg);
                break;
            }
        }));

    // We said we were not a terminal.
    return MacroAssembler::Jump();
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

