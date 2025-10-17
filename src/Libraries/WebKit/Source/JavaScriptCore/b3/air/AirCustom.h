/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 17, 2023.
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

#if ENABLE(B3_JIT)

#include "AirCCallingConvention.h"
#include "AirCode.h"
#include "AirGenerationContext.h"
#include "AirInst.h"
#include "AirSpecial.h"
#include "B3ValueInlines.h"
#include "B3WasmBoundsCheckValue.h"
#include "MacroAssembler.h"

namespace JSC { namespace B3 { namespace Air {

// This defines the behavior of custom instructions - i.e. those whose behavior cannot be
// described using AirOpcode.opcodes. If you define an opcode as "custom Foo" in that file, then
// you will need to create a "struct FooCustom" here that implements the custom behavior
// methods.
//
// The customizability granted by the custom instruction mechanism is strictly less than what
// you get using the Patch instruction and implementing a Special. However, that path requires
// allocating a Special object and ensuring that it's the first operand. For many instructions,
// that is not as convenient as using Custom, which makes the instruction look like any other
// instruction. Note that both of those extra powers of the Patch instruction happen because we
// special-case that instruction in many phases and analyses. Non-special-cased behaviors of
// Patch are implemented using the custom instruction mechanism.
//
// Specials are still more flexible if you need to list extra clobbered registers and you'd like
// that to be expressed as a bitvector rather than an arglist. They are also more flexible if
// you need to carry extra state around with the instruction. Also, Specials mean that you
// always have access to Code& even in methods that don't take a GenerationContext.

// Definition of Patch instruction. Patch is used to delegate the behavior of the instruction to the
// Special object, which will be the first argument to the instruction.
struct PatchCustom {
    static void forEachArg(Inst& inst, ScopedLambda<Inst::EachArgCallback> lambda)
    {
        // This is basically bogus, but it works for analyses that model Special as an
        // immediate.
        lambda(inst.args[0], Arg::Use, GP, pointerWidth());
        
        inst.args[0].special()->forEachArg(inst, lambda);
    }

    template<typename... Arguments>
    static bool isValidFormStatic(Arguments...)
    {
        return false;
    }

    static bool isValidForm(Inst& inst);

    static bool admitsStack(Inst& inst, unsigned argIndex)
    {
        if (!argIndex)
            return false;
        return inst.args[0].special()->admitsStack(inst, argIndex);
    }

    static bool admitsExtendedOffsetAddr(Inst& inst, unsigned argIndex)
    {
        if (!argIndex)
            return false;
        return inst.args[0].special()->admitsExtendedOffsetAddr(inst, argIndex);
    }

    static std::optional<unsigned> shouldTryAliasingDef(Inst& inst)
    {
        return inst.args[0].special()->shouldTryAliasingDef(inst);
    }
    
    static bool isTerminal(Inst& inst)
    {
        return inst.args[0].special()->isTerminal(inst);
    }

    static bool hasNonArgEffects(Inst& inst)
    {
        return inst.args[0].special()->hasNonArgEffects(inst);
    }

    static bool hasNonArgNonControlEffects(Inst& inst)
    {
        return inst.args[0].special()->hasNonArgNonControlEffects(inst);
    }

    static MacroAssembler::Jump generate(
        Inst& inst, CCallHelpers& jit, GenerationContext& context)
    {
        return inst.args[0].special()->generate(inst, jit, context);
    }
};

template<typename Subtype>
struct CommonCustomBase {
    static bool hasNonArgEffects(Inst& inst)
    {
        return Subtype::isTerminal(inst) || Subtype::hasNonArgNonControlEffects(inst);
    }
};

// Definition of CCall instruction. CCall is used for hot path C function calls. It's lowered to a
// Patch with an Air CCallSpecial along with code to marshal instructions. The lowering happens
// before register allocation, so that the register allocator sees the clobbers.
struct CCallCustom : public CommonCustomBase<CCallCustom> {
    template<typename Functor>
    static void forEachArg(Inst& inst, const Functor& functor)
    {
        CCallValue* value = inst.origin->as<CCallValue>();

        Code& code = inst.args[0].special()->code();

        // Skip the CCallSpecial Arg.
        unsigned index = 1;

        auto next = [&](Arg::Role role, Bank bank, Width width) {
            functor(inst.args[index++], role, bank, width);
        };

        next(Arg::Use, GP, pointerWidth()); // callee

        size_t resultCount = cCallResultCount(code, value);
        for (size_t n = 0; n < resultCount; ++n) {
            Type type = value->type().isTuple() ? code.proc().typeAtOffset(value->type(), n) : value->type();
            next(
                Arg::Def,
                bankForType(type),
                cCallArgumentRegisterWidth(type)
            );
        }

        for (unsigned i = 1; i < value->numChildren(); ++i) {
            Value* child = value->child(i);
            for (size_t j = 0; j < cCallArgumentRegisterCount(child->type()); j++) {
                next(
                    Arg::Use,
                    bankForType(child->type()),
                    cCallArgumentRegisterWidth(child->type())
                );
            }
        }
        ASSERT(index == inst.args.size());
    }

    template<typename... Arguments>
    static bool isValidFormStatic(Arguments...)
    {
        return false;
    }

    static bool isValidForm(Inst&);

    static bool admitsStack(Inst&, unsigned)
    {
        return true;
    }

    static bool admitsExtendedOffsetAddr(Inst&, unsigned)
    {
        return false;
    }
    
    static bool isTerminal(Inst&)
    {
        return false;
    }

    static bool hasNonArgNonControlEffects(Inst&)
    {
        return true;
    }

    // This just crashes, since we expect C calls to be lowered before generation.
    static MacroAssembler::Jump generate(Inst&, CCallHelpers&, GenerationContext&);
};

struct ColdCCallCustom : CCallCustom {
    template<typename Functor>
    static void forEachArg(Inst& inst, const Functor& functor)
    {
        // This is just like a call, but uses become cold.
        CCallCustom::forEachArg(
            inst,
            [&] (Arg& arg, Arg::Role role, Bank bank, Width width) {
                functor(arg, Arg::cooled(role), bank, width);
            });
    }
};

struct ShuffleCustom : public CommonCustomBase<ShuffleCustom> {
    template<typename Functor>
    static void forEachArg(Inst& inst, const Functor& functor)
    {
        unsigned limit = inst.args.size() / 3 * 3;
        for (unsigned i = 0; i < limit; i += 3) {
            Arg& src = inst.args[i + 0];
            Arg& dst = inst.args[i + 1];
            Arg& widthArg = inst.args[i + 2];
            Width width = widthArg.width();
            Bank bank = src.isGP() && dst.isGP() ? GP : FP;
            functor(src, Arg::Use, bank, width);
            functor(dst, Arg::Def, bank, width);
            functor(widthArg, Arg::Use, GP, Width8);
        }
    }

    template<typename... Arguments>
    static bool isValidFormStatic(Arguments...)
    {
        return false;
    }

    static bool isValidForm(Inst&);
    
    static bool admitsStack(Inst&, unsigned index)
    {
        switch (index % 3) {
        case 0:
        case 1:
            return true;
        default:
            return false;
        }
    }

    static bool admitsExtendedOffsetAddr(Inst&, unsigned)
    {
        return false;
    }

    static bool isTerminal(Inst&)
    {
        return false;
    }

    static bool hasNonArgNonControlEffects(Inst&)
    {
        return false;
    }

    static MacroAssembler::Jump generate(Inst&, CCallHelpers&, GenerationContext&);
};

struct EntrySwitchCustom : public CommonCustomBase<EntrySwitchCustom> {
    template<typename Func>
    static void forEachArg(Inst&, const Func&)
    {
    }
    
    template<typename... Arguments>
    static bool isValidFormStatic(Arguments...)
    {
        return !sizeof...(Arguments);
    }
    
    static bool isValidForm(Inst& inst)
    {
        return inst.args.isEmpty();
    }
    
    static bool admitsStack(Inst&, unsigned)
    {
        return false;
    }

    static bool admitsExtendedOffsetAddr(Inst&, unsigned)
    {
        return false;
    }
    
    static bool isTerminal(Inst&)
    {
        return true;
    }
    
    static bool hasNonArgNonControlEffects(Inst&)
    {
        return false;
    }

    static MacroAssembler::Jump generate(Inst&, CCallHelpers&, GenerationContext&)
    {
        // This should never be reached because we should have lowered EntrySwitch before
        // generation.
        UNREACHABLE_FOR_PLATFORM();
        return MacroAssembler::Jump();
    }
};

struct WasmBoundsCheckCustom : public CommonCustomBase<WasmBoundsCheckCustom> {
    template<typename Func>
    static void forEachArg(Inst& inst, const Func& functor)
    {
        functor(inst.args[0], Arg::Use, GP, pointerWidth());
        functor(inst.args[1], Arg::Use, GP, pointerWidth());
    }

    template<typename... Arguments>
    static bool isValidFormStatic(Arguments...)
    {
        return false;
    }

    static bool isValidForm(Inst&);

    static bool admitsStack(Inst&, unsigned)
    {
        return false;
    }

    static bool admitsExtendedOffsetAddr(Inst&, unsigned)
    {
        return false;
    }

    static bool isTerminal(Inst&)
    {
        return false;
    }
    
    static bool hasNonArgNonControlEffects(Inst&)
    {
        return true;
    }

    static MacroAssembler::Jump generate(Inst&, CCallHelpers&, GenerationContext&);
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
