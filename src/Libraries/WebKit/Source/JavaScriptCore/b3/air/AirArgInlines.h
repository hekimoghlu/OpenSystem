/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 1, 2023.
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

#include "AirArg.h"

namespace JSC { namespace B3 { namespace Air {

template<typename T> struct ArgThingHelper;

template<> struct ArgThingHelper<Tmp> {
    static bool is(const Arg& arg)
    {
        return arg.isTmp();
    }

    static Tmp as(const Arg& arg)
    {
        if (is(arg))
            return arg.tmp();
        return Tmp();
    }

    template<typename Functor>
    static void forEachFast(Arg& arg, const Functor& functor)
    {
        arg.forEachTmpFast(functor);
    }

    template<typename Functor>
    static void forEach(Arg& arg, Arg::Role role, Bank bank, Width width, const Functor& functor)
    {
        arg.forEachTmp(role, bank, width, functor);
    }
};

template<> struct ArgThingHelper<Arg> {
    static bool is(const Arg&)
    {
        return true;
    }

    static Arg as(const Arg& arg)
    {
        return arg;
    }

    template<typename Functor>
    static void forEachFast(Arg& arg, const Functor& functor)
    {
        functor(arg);
    }

    template<typename Functor>
    static void forEach(Arg& arg, Arg::Role role, Bank bank, Width width, const Functor& functor)
    {
        functor(arg, role, bank, width);
    }
};

template<> struct ArgThingHelper<StackSlot*> {
    static bool is(const Arg& arg)
    {
        return arg.isStack();
    }
    
    static StackSlot* as(const Arg& arg)
    {
        return arg.stackSlot();
    }
    
    template<typename Functor>
    static void forEachFast(Arg& arg, const Functor& functor)
    {
        if (!arg.isStack())
            return;
        
        StackSlot* stackSlot = arg.stackSlot();
        functor(stackSlot);
        arg = Arg::stack(stackSlot, arg.offset());
    }
    
    template<typename Functor>
    static void forEach(Arg& arg, Arg::Role role, Bank bank, Width width, const Functor& functor)
    {
        if (!arg.isStack())
            return;
        
        StackSlot* stackSlot = arg.stackSlot();
        
        // FIXME: This is way too optimistic about the meaning of "Def". It gets lucky for
        // now because our only use of "Anonymous" stack slots happens to want the optimistic
        // semantics. We could fix this by just changing the comments that describe the
        // semantics of "Anonymous".
        // https://bugs.webkit.org/show_bug.cgi?id=151128
        
        functor(stackSlot, role, bank, width);
        arg = Arg::stack(stackSlot, arg.offset());
    }
};

template<> struct ArgThingHelper<Reg> {
    static bool is(const Arg& arg)
    {
        return arg.isReg();
    }
    
    static Reg as(const Arg& arg)
    {
        return arg.reg();
    }
    
    template<typename Functor>
    static void forEachFast(Arg& arg, const Functor& functor)
    {
        arg.forEachTmpFast(
            [&] (Tmp& tmp) {
                if (!tmp.isReg())
                    return;
                
                Reg reg = tmp.reg();
                functor(reg);
                tmp = Tmp(reg);
            });
    }
    
    template<typename Functor>
    static void forEach(Arg& arg, Arg::Role argRole, Bank argBank, Width argWidth, const Functor& functor)
    {
        arg.forEachTmp(
            argRole, argBank, argWidth,
            [&] (Tmp& tmp, Arg::Role role, Bank bank, Width width) {
                if (!tmp.isReg())
                    return;
                
                Reg reg = tmp.reg();
                functor(reg, role, bank, width);
                tmp = Tmp(reg);
            });
    }
};

template<typename Thing>
bool Arg::is() const
{
    return ArgThingHelper<Thing>::is(*this);
}

template<typename Thing>
Thing Arg::as() const
{
    return ArgThingHelper<Thing>::as(*this);
}

template<typename Thing, typename Functor>
void Arg::forEachFast(const Functor& functor)
{
    ArgThingHelper<Thing>::forEachFast(*this, functor);
}

template<typename Thing, typename Functor>
void Arg::forEach(Role role, Bank bank, Width width, const Functor& functor)
{
    ArgThingHelper<Thing>::forEach(*this, role, bank, width, functor);
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
