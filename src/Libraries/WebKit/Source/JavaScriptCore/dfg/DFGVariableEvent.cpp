/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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
#include "DFGVariableEvent.h"

#if ENABLE(DFG_JIT)

#include "FPRInfo.h"
#include "GPRInfo.h"
#include "OperandsInlines.h"

namespace JSC { namespace DFG {

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

void VariableEvent::dump(PrintStream& out) const
{
    switch (kind()) {
    case Reset:
        out.printf("Reset");
        break;
    case BirthToFill:
        dumpFillInfo("BirthToFill", out);
        break;
    case BirthToSpill:
        dumpSpillInfo("BirthToSpill", out);
        break;
    case Birth:
        out.print("Birth(", id(), ")");
        break;
    case Fill:
        dumpFillInfo("Fill", out);
        break;
    case Spill:
        dumpSpillInfo("Spill", out);
        break;
    case Death:
        out.print("Death(", id(), ")");
        break;
    case MovHintEvent:
        out.print("MovHint(", id(), ", ", operand(), ")");
        break;
    case SetLocalEvent:
        out.print(
            "SetLocal(machine:", machineRegister(), " -> bytecode:", operand(),
            ", ", dataFormatToString(dataFormat()), ")");
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }
}

void VariableEvent::dumpFillInfo(const char* name, PrintStream& out) const
{
    out.print(name, "(", id(), ", ");
    if (dataFormat() == DataFormatDouble)
        out.printf("%s", FPRInfo::debugName(fpr()).characters());
#if USE(JSVALUE32_64)
    else if (dataFormat() & DataFormatJS)
        out.printf("%s:%s", GPRInfo::debugName(tagGPR()).characters(), GPRInfo::debugName(payloadGPR()).characters());
#endif
    else
        out.printf("%s", GPRInfo::debugName(gpr()).characters());
    out.printf(", %s)", dataFormatToString(dataFormat()));
}

void VariableEvent::dumpSpillInfo(const char* name, PrintStream& out) const
{
    out.print(name, "(", id(), ", ", spillRegister(), ", ", dataFormatToString(dataFormat()), ")");
}

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

