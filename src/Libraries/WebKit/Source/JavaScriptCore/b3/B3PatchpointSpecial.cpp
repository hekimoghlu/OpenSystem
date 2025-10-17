/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 12, 2021.
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
#include "B3PatchpointSpecial.h"

#if ENABLE(B3_JIT)

#include "AirCode.h"
#include "AirGenerationContext.h"
#include "B3ProcedureInlines.h"
#include "B3StackmapGenerationParams.h"
#include "B3ValueInlines.h"

#include <wtf/ListDump.h>

namespace JSC { namespace B3 {

using Arg = Air::Arg;
using Inst = Air::Inst;

PatchpointSpecial::PatchpointSpecial() = default;

PatchpointSpecial::~PatchpointSpecial() = default;

void PatchpointSpecial::forEachArg(Inst& inst, const ScopedLambda<Inst::EachArgCallback>& callback)
{
    const Procedure& procedure = code().proc();
    PatchpointValue* patchpoint = inst.origin->as<PatchpointValue>();
    unsigned argIndex = 1;

    Type type = patchpoint->type();
    for (; argIndex <= procedure.resultCount(type); ++argIndex) {
        Arg::Role role;
        if (patchpoint->resultConstraints[argIndex - 1].kind() == ValueRep::SomeEarlyRegister)
            role = Arg::EarlyDef;
        else
            role = Arg::Def;

        Type argType = type.isTuple() ? procedure.extractFromTuple(type, argIndex - 1) : type;
        callback(inst.args[argIndex], role, bankForType(argType), widthForType(argType));
    }

    forEachArgImpl(0, argIndex, inst, SameAsRep, std::nullopt, callback, std::nullopt);
    argIndex += inst.origin->numChildren();

    for (unsigned i = patchpoint->numGPScratchRegisters; i--;)
        callback(inst.args[argIndex++], Arg::Scratch, GP, conservativeWidth(GP));
    for (unsigned i = patchpoint->numFPScratchRegisters; i--;)
        callback(inst.args[argIndex++], Arg::Scratch, FP, procedure.usesSIMD() ? conservativeWidth(FP) : conservativeWidthWithoutVectors(FP));
}

bool PatchpointSpecial::isValid(Inst& inst)
{
    const Procedure& procedure = code().proc();
    PatchpointValue* patchpoint = inst.origin->as<PatchpointValue>();
    unsigned argIndex = 1;

    Type type = patchpoint->type();
    for (; argIndex <= procedure.resultCount(type); ++argIndex) {
        if (argIndex >= inst.args.size())
            return false;
        
        if (!isArgValidForType(inst.args[argIndex], type.isTuple() ? procedure.extractFromTuple(type, argIndex - 1) : type))
            return false;
        if (!isArgValidForRep(code(), inst.args[argIndex], patchpoint->resultConstraints[argIndex - 1]))
            return false;
    }

    if (!isValidImpl(0, argIndex, inst))
        return false;
    argIndex += patchpoint->numChildren();

    if (argIndex + patchpoint->numGPScratchRegisters + patchpoint->numFPScratchRegisters
        != inst.args.size())
        return false;

    for (unsigned i = patchpoint->numGPScratchRegisters; i--;) {
        Arg arg = inst.args[argIndex++];
        if (!arg.isGPTmp())
            return false;
    }
    for (unsigned i = patchpoint->numFPScratchRegisters; i--;) {
        Arg arg = inst.args[argIndex++];
        if (!arg.isFPTmp())
            return false;
    }

    return true;
}

bool PatchpointSpecial::admitsStack(Inst& inst, unsigned argIndex)
{
    ASSERT(argIndex);

    Type type = inst.origin->type();
    unsigned returnCount = code().proc().resultCount(type);

    if (argIndex <= returnCount) {
        switch (inst.origin->as<PatchpointValue>()->resultConstraints[argIndex - 1].kind()) {
        case ValueRep::WarmAny:
        case ValueRep::StackArgument:
            return true;
        case ValueRep::SomeRegister:
        case ValueRep::SomeRegisterWithClobber:
        case ValueRep::SomeEarlyRegister:
        case ValueRep::SomeLateRegister:
        case ValueRep::Register:
        case ValueRep::LateRegister:
            return false;
#if USE(JSVALUE32_64)
        case ValueRep::RegisterPair:
#endif
        default:
            RELEASE_ASSERT_NOT_REACHED();
            return false;
        }
    }

    return admitsStackImpl(0, returnCount + 1, inst, argIndex);
}

bool PatchpointSpecial::admitsExtendedOffsetAddr(Inst& inst, unsigned argIndex)
{
    return admitsStack(inst, argIndex);
}

MacroAssembler::Jump PatchpointSpecial::generate(Inst& inst, CCallHelpers& jit, Air::GenerationContext& context)
{
    const Procedure& procedure = code().proc();
    PatchpointValue* value = inst.origin->as<PatchpointValue>();
    ASSERT(value);

    Vector<ValueRep> reps;
    unsigned offset = 1;

    Type type = value->type();
    while (offset <= procedure.resultCount(type))
        reps.append(repForArg(*context.code, inst.args[offset++]));
    reps.appendVector(repsImpl(context, 0, offset, inst));
    offset += value->numChildren();

    StackmapGenerationParams params(value, reps, context);

    for (unsigned i = value->numGPScratchRegisters; i--;)
        params.m_gpScratch.append(inst.args[offset++].gpr());
    for (unsigned i = value->numFPScratchRegisters; i--;)
        params.m_fpScratch.append(inst.args[offset++].fpr());
    
    value->m_generator->run(jit, params);

    return MacroAssembler::Jump();
}

bool PatchpointSpecial::isTerminal(Inst& inst)
{
    return inst.origin->as<PatchpointValue>()->effects.terminal;
}

void PatchpointSpecial::dumpImpl(PrintStream& out) const
{
    out.print("Patchpoint");
}

void PatchpointSpecial::deepDumpImpl(PrintStream& out) const
{
    out.print("Lowered B3::PatchpointValue.");
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)
