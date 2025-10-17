/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 27, 2025.
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
#include "AirInst.h"

#if ENABLE(B3_JIT)

#include "AirInstInlines.h"
#include <wtf/ListDump.h>

namespace JSC { namespace B3 { namespace Air {

bool Inst::hasEarlyDef()
{
    if (kind.opcode == Patch && !extraEarlyClobberedRegs().isEmpty())
        return true;
    bool result = false;
    forEachArg(
        [&] (Arg&, Arg::Role role, Bank, Width) {
            result |= Arg::isEarlyDef(role);
        });
    return result;
}

bool Inst::hasLateUseOrDef()
{
    if (kind.opcode == Patch && !extraClobberedRegs().isEmpty())
        return true;
    bool result = false;
    forEachArg(
        [&] (Arg&, Arg::Role role, Bank, Width) {
            result |= Arg::isLateUse(role) || Arg::isLateDef(role);
        });
    return result;
}

bool Inst::needsPadding(Inst* prevInst, Inst* nextInst)
{
    bool result = prevInst && nextInst && prevInst->hasLateUseOrDef() && nextInst->hasEarlyDef();
    return result;
}

bool Inst::hasArgEffects()
{
    bool result = false;
    forEachArg(
        [&] (Arg&, Arg::Role role, Bank, Width) {
            if (Arg::isAnyDef(role))
                result = true;
        });
    return result;
}

unsigned Inst::jsHash() const
{
    // FIXME: This should do something for flags.
    // https://bugs.webkit.org/show_bug.cgi?id=162751
    unsigned result = static_cast<unsigned>(kind.opcode);
    
    for (const Arg& arg : args)
        result += arg.jsHash();
    
    return result;
}

void Inst::dump(PrintStream& out) const
{
    out.print(kind, " ", listDump(args));
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
