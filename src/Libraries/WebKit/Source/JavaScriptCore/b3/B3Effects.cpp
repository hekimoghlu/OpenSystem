/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#include "B3Effects.h"

#if ENABLE(B3_JIT)

#include <wtf/CommaPrinter.h>

namespace JSC { namespace B3 {

namespace {

// These helpers cascade in such a way that after the helper for terminal, we don't have to worry
// about terminal again, since the terminal case considers all ways that a terminal may interfere
// with something else. And after the exit sideways case, we don't have to worry about either
// exitsSideways or terminal. And so on...

bool interferesWithTerminal(const Effects& terminal, const Effects& other)
{
    if (!terminal.terminal)
        return false;
    return other.terminal || other.controlDependent || other.writesLocalState || other.writes || other.writesPinned;
}

bool interferesWithExitSideways(const Effects& exitsSideways, const Effects& other)
{
    if (!exitsSideways.exitsSideways)
        return false;
    return other.controlDependent || other.writes || other.writesPinned;
}

bool interferesWithWritesLocalState(const Effects& writesLocalState, const Effects& other)
{
    if (!writesLocalState.writesLocalState)
        return false;
    return other.writesLocalState || other.readsLocalState;
}

bool interferesWithWritesPinned(const Effects& writesPinned, const Effects& other)
{
    if (!writesPinned.writesPinned)
        return false;
    return other.writesPinned || other.readsPinned;
}

} // anonymous namespace

bool Effects::interferes(const Effects& other) const
{
    return interferesWithTerminal(*this, other)
        || interferesWithTerminal(other, *this)
        || interferesWithExitSideways(*this, other)
        || interferesWithExitSideways(other, *this)
        || interferesWithWritesLocalState(*this, other)
        || interferesWithWritesLocalState(other, *this)
        || interferesWithWritesPinned(*this, other)
        || interferesWithWritesPinned(other, *this)
        || writes.overlaps(other.writes)
        || writes.overlaps(other.reads)
        || reads.overlaps(other.writes)
        || (fence && other.fence);
}

void Effects::dump(PrintStream& out) const
{
    CommaPrinter comma("|"_s);
    if (terminal)
        out.print(comma, "Terminal"_s);
    if (exitsSideways)
        out.print(comma, "ExitsSideways"_s);
    if (controlDependent)
        out.print(comma, "ControlDependent"_s);
    if (writesLocalState)
        out.print(comma, "WritesLocalState"_s);
    if (readsLocalState)
        out.print(comma, "ReadsLocalState"_s);
    if (writesPinned)
        out.print(comma, "WritesPinned"_s);
    if (readsPinned)
        out.print(comma, "ReadsPinned"_s);
    if (fence)
        out.print(comma, "Fence"_s);
    if (writes)
        out.print(comma, "Writes:"_s, writes);
    if (reads)
        out.print(comma, "Reads:"_s, reads);
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

