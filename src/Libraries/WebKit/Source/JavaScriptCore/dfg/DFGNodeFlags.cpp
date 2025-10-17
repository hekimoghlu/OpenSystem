/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#include "DFGNodeFlags.h"

#if ENABLE(DFG_JIT)

#include <wtf/CommaPrinter.h>
#include <wtf/StringPrintStream.h>

namespace JSC { namespace DFG {

void dumpNodeFlags(PrintStream& actualOut, NodeFlags flags)
{
    StringPrintStream out;
    CommaPrinter comma("|"_s);
    
    if (flags & NodeResultMask) {
        switch (flags & NodeResultMask) {
        case NodeResultJS:
            out.print(comma, "JS"_s);
            break;
        case NodeResultNumber:
            out.print(comma, "Number"_s);
            break;
        case NodeResultDouble:
            out.print(comma, "Double"_s);
            break;
        case NodeResultInt32:
            out.print(comma, "Int32"_s);
            break;
        case NodeResultInt52:
            out.print(comma, "Int52"_s);
            break;
        case NodeResultBoolean:
            out.print(comma, "Boolean"_s);
            break;
        case NodeResultStorage:
            out.print(comma, "Storage"_s);
            break;
        default:
            RELEASE_ASSERT_NOT_REACHED();
            break;
        }
    }
    
    if (flags & NodeMustGenerate)
        out.print(comma, "MustGen"_s);
    
    if (flags & NodeHasVarArgs)
        out.print(comma, "VarArgs"_s);
    
    if (flags & NodeResultMask) {
        if (!(flags & NodeBytecodeUsesAsNumber))
            out.print(comma, "PureInt"_s);
        else
            out.print(comma, "PureNum"_s);
        if (flags & NodeBytecodeNeedsNegZero)
            out.print(comma, "NeedsNegZero"_s);
        if (flags & NodeBytecodeNeedsNaNOrInfinity)
            out.print(comma, "NeedsNaNOrInfinity"_s);
        if (flags & NodeBytecodeUsesAsOther)
            out.print(comma, "UseAsOther"_s);
    }

    if (flags & NodeMayHaveDoubleResult)
        out.print(comma, "MayHaveDoubleResult"_s);

    if (flags & NodeMayHaveBigInt32Result)
        out.print(comma, "MayHaveBigInt32Result"_s);

    if (flags & NodeMayHaveHeapBigIntResult)
        out.print(comma, "MayHaveHeapBigIntResult"_s);

    if (flags & NodeMayHaveNonNumericResult)
        out.print(comma, "MayHaveNonNumericResult"_s);

    if (flags & NodeMayOverflowInt52)
        out.print(comma, "MayOverflowInt52"_s);

    if (flags & NodeMayOverflowInt32InBaseline)
        out.print(comma, "MayOverflowInt32InBaseline"_s);

    if (flags & NodeMayOverflowInt32InDFG)
        out.print(comma, "MayOverflowInt32InDFG"_s);

    if (flags & NodeMayNegZeroInBaseline)
        out.print(comma, "MayNegZeroInBaseline"_s);
    
    if (flags & NodeMayNegZeroInDFG)
        out.print(comma, "MayNegZeroInDFG"_s);
    
    if (flags & NodeBytecodeUsesAsInt)
        out.print(comma, "UseAsInt"_s);

    if (flags & NodeBytecodePrefersArrayIndex)
        out.print(comma, "ReallyWantsInt"_s);
    
    if (flags & NodeIsFlushed)
        out.print(comma, "IsFlushed"_s);
    
    CString string = out.toCString();
    if (!string.length())
        actualOut.print("<empty>"_s);
    else
        actualOut.print(string);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

