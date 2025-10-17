/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 2, 2021.
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
#include "AirInst.h"
#include <wtf/Vector.h>

namespace JSC { namespace B3 {

class Value;

namespace Air {

class Code;

inline Opcode moveFor(Bank bank, Width width)
{
    switch (width) {
    case Width32:
        return bank == GP ? Move32 : MoveFloat;
    case Width64:
        return bank == GP ? Move : MoveDouble;
    case Width128:
        ASSERT(bank == FP);
        return MoveVector;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return Oops;
    }
}

class ShufflePair {
public:
    ShufflePair()
    {
    }
    
    ShufflePair(const Arg& src, const Arg& dst, Width width)
        : m_src(src)
        , m_dst(dst)
        , m_width(width)
    {
    }

    const Arg& src() const { return m_src; }
    const Arg& dst() const { return m_dst; }

    // The width determines the kind of move we do. You can only choose Width32 or Width64 right now.
    // For GP, it picks between Move32 and Move. For FP, it picks between MoveFloat and MoveDouble.
    Width width() const { return m_width; }
    
    Bank bank() const;

    // Creates an instruction sequence for the move represented by this shuffle pair.
    // You need to pass Code because we may need to create a tmp.
    Vector<Inst, 2> insts(Code&, Value* origin) const;

    void dump(PrintStream&) const;
    
private:
    Arg m_src;
    Arg m_dst;
    Width m_width { Width8 };
};

// Create a Shuffle instruction.
Inst createShuffle(Value* origin, const Vector<ShufflePair>&);

// Perform a shuffle of a given type. The scratch argument is mandatory. You should pass it as
// follows: If you know that you have scratch registers or temporaries available - that is, they're
// registers that are not mentioned in the shuffle, have the same type as the shuffle, and are not
// live at the shuffle - then you can pass them. If you don't have scratch registers available or if
// you don't feel like looking for them, you can pass memory locations. It's always safe to pass a
// pair of memory locations, and replacing either memory location with a register can be viewed as an
// optimization. It's a pretty important optimization. Some more notes:
//
// - We define scratch registers as things that are not live before the shuffle and are not one of
//   the destinations of the shuffle. Not being live before the shuffle also means that they cannot
//   be used for any of the sources of the shuffle.
//
// - A second scratch location is only needed when you have shuffle pairs where memory is used both
//   as source and destination.
//
// - You're guaranteed not to need any scratch locations if there is a Swap instruction available for
//   the type and you don't have any memory locations that are both the source and the destination of
//   some pairs. GP supports Swap on x86 while FP never supports Swap.
//
// - Passing memory locations as scratch if are running emitShuffle() before register allocation is
//   silly, since that will cause emitShuffle() to pick some specific registers when it does need
//   scratch. One easy way to avoid that predicament is to ensure that you call emitShuffle() after
//   register allocation. For this reason we could add a Shuffle instruction so that we can defer
//   shufflings until after regalloc.
//
// - Shuffles with memory=>memory pairs are not very well tuned. You should avoid them if you want
//   performance. If you need to do them, then making sure that you reserve a temporary is one way to
//   get acceptable performance.
//
// NOTE: Use this method (and its friend below) to emit shuffles after register allocation. Before
// register allocation it is much better to simply use the Shuffle instruction.
Vector<Inst> emitShuffle(
    Code& code, Vector<ShufflePair>, std::array<Arg, 2> scratch, Bank, Value* origin);

// Perform a shuffle that involves any number of types. Pass scratch registers or memory locations
// for each type according to the rules above.
Vector<Inst> emitShuffle(
    Code& code, const Vector<ShufflePair>&,
    const std::array<Arg, 2>& gpScratch, const std::array<Arg, 2>& fpScratch,
    Value* origin);

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
