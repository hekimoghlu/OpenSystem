/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#include "WasmOpcodeOrigin.h"

#include <wtf/text/MakeString.h>

#if ENABLE(WEBASSEMBLY_OMGJIT)

namespace JSC { namespace Wasm {

void OpcodeOrigin::dump(PrintStream& out) const
{
    switch (opcode()) {
#if USE(JSVALUE64)
    case OpType::ExtGC:
        out.print("{opcode: ", makeString(gcOpcode()), ", location: ", RawHex(location()), "}");
        break;
    case OpType::Ext1:
        out.print("{opcode: ", makeString(ext1Opcode()), ", location: ", RawHex(location()), "}");
        break;
    case OpType::ExtSIMD:
        out.print("{opcode: ", makeString(simdOpcode()), ", location: ", RawHex(location()), "}");
        break;
    case OpType::ExtAtomic:
        out.print("{opcode: ", makeString(atomicOpcode()), ", location: ", RawHex(location()), "}");
        break;
#endif
    default:
        out.print("{opcode: ", makeString(opcode()), ", location: ", RawHex(location()), "}");
        break;
    }
}

} } // namespace JSC::Wasm

#endif // ENABLE(WEBASSEMBLY_OMGJIT)
