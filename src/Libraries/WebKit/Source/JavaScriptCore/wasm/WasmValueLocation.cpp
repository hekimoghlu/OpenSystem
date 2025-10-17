/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#include "WasmValueLocation.h"

#if ENABLE(WEBASSEMBLY)

namespace JSC { namespace Wasm {

void ValueLocation::dump(PrintStream& out) const
{
    out.print(m_kind);
    switch (m_kind) {
    case GPRRegister:
        out.print("(", jsr(), ")");
        return;
    case FPRRegister:
        out.print("(", fpr(), ")");
        return;
    case Stack:
        out.print("(", offsetFromFP(), ")");
        return;
    case StackArgument:
        out.print("(", offsetFromSP(), ")");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} } // namespace JSC::Wasm

namespace WTF {

using namespace JSC::Wasm;

void printInternal(PrintStream& out, ValueLocation::Kind kind)
{
    switch (kind) {
    case ValueLocation::GPRRegister:
        out.print("GPRRegister");
        return;
    case ValueLocation::FPRRegister:
        out.print("FPRRegister");
        return;
    case ValueLocation::Stack:
        out.print("Stack");
        return;
    case ValueLocation::StackArgument:
        out.print("StackArgument");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(WEBASSEMBLY)
