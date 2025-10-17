/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 26, 2023.
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

#include "B3Effects.h"
#include "B3StackmapValue.h"
#include "B3Value.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace JSC { namespace B3 {

class PatchpointValue final : public StackmapValue {
public:
    typedef StackmapValue Base;
    
    static bool accepts(Kind kind) { return kind.opcode() == Patchpoint; }

    ~PatchpointValue() final;

    // The effects of the patchpoint. This defaults to Effects::forCall(), but you can set it to anything.
    //
    // If there are no effects, B3 is free to assume any use of this PatchpointValue can be replaced with
    // a use of a different PatchpointValue, so long as the other one also has no effects and has the
    // same children. Note that this comparison ignores child constraints, the result constraint, and all
    // other StackmapValue meta-data. If there are read effects but not write effects, then this same sort
    // of substitution could be made so long as there are no interfering writes.
    Effects effects;

    // The input representation (i.e. constraint) of the return value. This defaults to WarmAny if the
    // type is Void and it defaults to SomeRegister otherwise. It's illegal to mess with this if the type
    // is Void. Otherwise you can set this to any input constraint. If the type of the patchpoint is a tuple
    // the constrants must be set explicitly.
    Vector<ValueRep, 1> resultConstraints;

    // The number of scratch registers that this patchpoint gets. The scratch register is guaranteed
    // to be different from any input register and the destination register. It's also guaranteed not
    // to be clobbered either early or late. These are 0 by default.
    uint8_t numGPScratchRegisters { 0 };
    uint8_t numFPScratchRegisters { 0 };

    B3_SPECIALIZE_VALUE_FOR_FINAL_SIZE_VARARGS_CHILDREN

private:
    void dumpMeta(CommaPrinter&, PrintStream&) const final;

    friend class Procedure;
    friend class Value;

    static Opcode opcodeFromConstructor(Type, Origin) { return Patchpoint; }
    static Opcode opcodeFromConstructor(Type, Origin, Kind) { return Patchpoint; }
    JS_EXPORT_PRIVATE PatchpointValue(Type, Origin, Kind = Patchpoint);
};

} } // namespace JSC::B3

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#endif // ENABLE(B3_JIT)
