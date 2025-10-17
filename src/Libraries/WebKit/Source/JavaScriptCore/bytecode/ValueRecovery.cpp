/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#include "ValueRecovery.h"

#include "JSCJSValueInlines.h"

namespace JSC {

JSValue ValueRecovery::recover(CallFrame* callFrame) const
{
    switch (technique()) {
    case DisplacedInJSStack:
        return callFrame->r(virtualRegister()).jsValue();
    case Int32DisplacedInJSStack:
        return jsNumber(callFrame->r(virtualRegister()).unboxedInt32());
    case Int52DisplacedInJSStack:
        return jsNumber(callFrame->r(virtualRegister()).unboxedInt52());
    case StrictInt52DisplacedInJSStack:
        return jsNumber(callFrame->r(virtualRegister()).unboxedStrictInt52());
    case DoubleDisplacedInJSStack:
        return jsNumber(purifyNaN(callFrame->r(virtualRegister()).unboxedDouble()));
    case CellDisplacedInJSStack:
        return callFrame->r(virtualRegister()).unboxedCell();
    case BooleanDisplacedInJSStack:
#if USE(JSVALUE64)
        return callFrame->r(virtualRegister()).jsValue();
#else
        return jsBoolean(callFrame->r(virtualRegister()).unboxedBoolean());
#endif
    case Constant:
        return constant();
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return JSValue();
    }
}

#if ENABLE(JIT)

void ValueRecovery::dumpInContext(PrintStream& out, DumpContext* context) const
{
    switch (technique()) {
    case InGPR:
        out.print(gpr());
        return;
    case UnboxedInt32InGPR:
        out.print("int32(", gpr(), ")");
        return;
    case UnboxedInt52InGPR:
        out.print("int52(", gpr(), ")");
        return;
    case UnboxedStrictInt52InGPR:
        out.print("strictInt52(", gpr(), ")");
        return;
    case UnboxedBooleanInGPR:
        out.print("bool(", gpr(), ")");
        return;
    case UnboxedCellInGPR:
        out.print("cell(", gpr(), ")");
        return;
    case InFPR:
        out.print(fpr());
        return;
    case UnboxedDoubleInFPR:
        out.print("double(", fpr(), ")");
        return;
#if USE(JSVALUE32_64)
    case InPair:
        out.print("pair(", tagGPR(), ", ", payloadGPR(), ")");
        return;
#endif
    case DisplacedInJSStack:
        out.print("*", virtualRegister());
        return;
    case Int32DisplacedInJSStack:
        out.print("*int32(", virtualRegister(), ")");
        return;
#if USE(JSVALUE32_64)
    case Int32TagDisplacedInJSStack:
        out.print("*int32Tag(", virtualRegister(), ")");
        return;
#endif
    case Int52DisplacedInJSStack:
        out.print("*int52(", virtualRegister(), ")");
        return;
    case StrictInt52DisplacedInJSStack:
        out.print("*strictInt52(", virtualRegister(), ")");
        return;
    case DoubleDisplacedInJSStack:
        out.print("*double(", virtualRegister(), ")");
        return;
    case CellDisplacedInJSStack:
        out.print("*cell(", virtualRegister(), ")");
        return;
    case BooleanDisplacedInJSStack:
        out.print("*bool(", virtualRegister(), ")");
        return;
    case DirectArgumentsThatWereNotCreated:
        out.print("DirectArguments(", nodeID(), ")");
        return;
    case ClonedArgumentsThatWereNotCreated:
        out.print("ClonedArguments(", nodeID(), ")");
        return;
    case Constant:
        out.print("[", inContext(constant(), context), "]");
        return;
    case DontKnow:
        out.printf("!");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void ValueRecovery::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}
#endif // ENABLE(JIT)

} // namespace JSC

