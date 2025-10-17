/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 8, 2022.
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
#ifndef AirKind_h
#define AirKind_h

#if ENABLE(B3_JIT)

#include "AirOpcode.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace B3 { namespace Air {

// Air opcodes are always carried around with some flags. These flags are understood as having no
// meaning if they are set for an opcode to which they do not apply. This makes sense, since Air
// is a complex instruction set and most of these flags can apply to basically any opcode. In
// fact, it's recommended to only represent something as a flag if you believe that it is largely
// opcode-agnostic.

struct Kind {
    Kind(Opcode opcode)
        : opcode(opcode)
    {
    }
    
    Kind()
        : Kind(Nop)
    {
    }
    
    friend bool operator==(const Kind&, const Kind&) = default;
    
    unsigned hash() const
    {
        return static_cast<unsigned>(opcode) + (static_cast<unsigned>(effects) << 16) + (static_cast<unsigned>(spill) << 17);
    }
    
    explicit operator bool() const
    {
        return *this != Kind();
    }
    
    void dump(PrintStream&) const;
    
    Opcode opcode;
    
    // This is an opcode-agnostic flag that indicates that we expect that this instruction will do
    // any of the following:
    // - Trap.
    // - Perform some non-arg non-control effect.
    bool effects : 1 { false };

    // This marks whether this instruction was generated for stack spilling.
    bool spill : 1 { false };
};

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

#endif // AirKind_h

