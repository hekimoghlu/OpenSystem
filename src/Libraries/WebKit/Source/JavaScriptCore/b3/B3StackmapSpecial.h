/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#include "AirSpecial.h"
#include "B3ValueRep.h"
#include <wtf/TZoneMalloc.h>

namespace JSC { namespace B3 {

namespace Air { class Code; }

// This is a base class for specials that have stackmaps. Note that it can find the Stackmap by
// asking for the Inst's origin. Hence, these objects don't need to even hold a reference to the
// Stackmap.

class StackmapSpecial : public Air::Special {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(StackmapSpecial, JS_EXPORT_PRIVATE);
public:
    StackmapSpecial();
    ~StackmapSpecial() override;

    enum RoleMode : int8_t {
        SameAsRep,
        ForceLateUseUnlessRecoverable,
        ForceLateUse
    };

protected:
    void reportUsedRegisters(Air::Inst&, const RegisterSetBuilder&) final;
    RegisterSetBuilder extraEarlyClobberedRegs(Air::Inst&) final;
    RegisterSetBuilder extraClobberedRegs(Air::Inst&) final;

    // Note that this does not override generate() or dumpImpl()/deepDumpImpl(). We have many
    // subclasses that implement that.
    void forEachArgImpl(
        unsigned numIgnoredB3Args, unsigned numIgnoredAirArgs,
        Air::Inst&, RoleMode, std::optional<unsigned> firstRecoverableIndex,
        const ScopedLambda<Air::Inst::EachArgCallback>&, std::optional<Width> optionalDefArgWidth);

    bool isValidImpl(
        unsigned numIgnoredB3Args, unsigned numIgnoredAirArgs,
        Air::Inst&);
    bool admitsStackImpl(
        unsigned numIgnoredB3Args, unsigned numIgnoredAirArgs,
        Air::Inst&, unsigned argIndex);

    // Appends the reps for the Inst's args, starting with numIgnoredArgs, to the given vector.
    Vector<ValueRep> repsImpl(
        Air::GenerationContext&, unsigned numIgnoredB3Args, unsigned numIgnoredAirArgs, Air::Inst&);

    static bool isArgValidForType(const Air::Arg&, Type);
    static bool isArgValidForRep(Air::Code&, const Air::Arg&, const ValueRep&);
    static ValueRep repForArg(Air::Code&, const Air::Arg&);
};

} } // namespace JSC::B3

namespace WTF {

void printInternal(PrintStream&, JSC::B3::StackmapSpecial::RoleMode);

} // namespace WTF

#endif // ENABLE(B3_JIT)
