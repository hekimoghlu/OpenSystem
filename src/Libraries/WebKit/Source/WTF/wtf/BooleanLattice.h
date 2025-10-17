/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 15, 2024.
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

#include <wtf/PrintStream.h>

namespace WTF {

// This is a boolean type that is part of an abstract value lattice. It's useful for inferring what
// the boolean value of something is by exploring all boolean values we encounter.
//
// It's useful to think of a lattice as a set. The comments below also describe what the enum values
// mean in terms of sets.
//
// FIXME: This would work a lot better as a class with methods. Then we could ensure that the default
// value is always Bottom, we could have nice conversions to and from boolean, and things like the
// leastUpperBound function could be a member function with a nicer name.
// https://bugs.webkit.org/show_bug.cgi?id=185804
enum class BooleanLattice : uint8_t {
    // Bottom means that we haven't seen any boolean values yet. We don't know what boolean value we
    // will infer yet. If we are left with Bottom after we have considered all booleans, it means
    // that we did not see any booleans.
    //
    // This represents the empty set.
    Bottom = 0,
    
    // We definitely saw false.
    //
    // This represents a set that just contains false.
    False = 1,

    // We definitely saw true.
    //
    // This represents a set that just contains true.
    True = 2,
    
    // Top means that we have seen both false and true. Like Bottom, it means that we don't know what
    // boolean value this lattice has. But unlike Bottom, which bases its lack of knowledge on not
    // having seen any booleans, Top bases its lack of knowledge based on having seen both False and
    // True.
    //
    // This represents a set that contains both false and true.
    Top = 3
};

inline BooleanLattice leastUpperBoundOfBooleanLattices(BooleanLattice a, BooleanLattice b)
{
    return static_cast<BooleanLattice>(static_cast<uint8_t>(a) | static_cast<uintptr_t>(b));
}

inline void printInternal(PrintStream& out, BooleanLattice value)
{
    switch (value) {
    case BooleanLattice::Bottom:
        out.print("Bottom");
        return;
    case BooleanLattice::False:
        out.print("False");
        return;
    case BooleanLattice::True:
        out.print("True");
        return;
    case BooleanLattice::Top:
        out.print("Top");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

using WTF::BooleanLattice;
using WTF::leastUpperBoundOfBooleanLattices;

