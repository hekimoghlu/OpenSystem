/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 10, 2025.
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
#include "DFGArithMode.h"

#if ENABLE(DFG_JIT)

#include "DFGOperations.h"
#include "JSCInlines.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

Arith::UnaryFunction arithUnaryFunction(Arith::UnaryType type)
{
    switch (type) {
#define DFG_ARITH_UNARY(capitalizedName, lowerName) \
    case Arith::UnaryType::capitalizedName: \
        return JSC::Math::lowerName##Double;
    FOR_EACH_ARITH_UNARY_OP(DFG_ARITH_UNARY)
#undef DFG_ARITH_UNARY
    }
    RELEASE_ASSERT_NOT_REACHED();

}

Arith::UnaryOperation arithUnaryOperation(Arith::UnaryType type)
{
    switch (type) {
#define DFG_ARITH_UNARY(capitalizedName, lowerName) \
    case Arith::UnaryType::capitalizedName: \
        return static_cast<Arith::UnaryOperation>(operationArith##capitalizedName);
    FOR_EACH_ARITH_UNARY_OP(DFG_ARITH_UNARY)
#undef DFG_ARITH_UNARY
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} } // namespace JSC::DFG

namespace WTF {

void printInternal(PrintStream& out, JSC::DFG::Arith::Mode mode)
{
    switch (mode) {
    case JSC::DFG::Arith::NotSet:
        out.print("NotSet");
        return;
    case JSC::DFG::Arith::Unchecked:
        out.print("Unchecked");
        return;
    case JSC::DFG::Arith::CheckOverflow:
        out.print("CheckOverflow");
        return;
    case JSC::DFG::Arith::CheckOverflowAndNegativeZero:
        out.print("CheckOverflowAndNegativeZero");
        return;
    case JSC::DFG::Arith::DoOverflow:
        out.print("DoOverflow");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, JSC::DFG::Arith::RoundingMode mode)
{
    switch (mode) {
    case JSC::DFG::Arith::RoundingMode::Int32:
        out.print("Int32");
        return;
    case JSC::DFG::Arith::RoundingMode::Int32WithNegativeZeroCheck:
        out.print("Int32WithNegativeZeroCheck");
        return;
    case JSC::DFG::Arith::RoundingMode::Double:
        out.print("Double");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void printInternal(PrintStream& out, JSC::DFG::Arith::UnaryType type)
{
    switch (type) {
#define DFG_ARITH_UNARY(capitalizedName, lowerName) \
    case JSC::DFG::Arith::UnaryType::capitalizedName: \
        out.print(#capitalizedName); \
        return;
    FOR_EACH_ARITH_UNARY_OP(DFG_ARITH_UNARY)
#undef DFG_ARITH_UNARY
    }
    RELEASE_ASSERT_NOT_REACHED();
}


} // namespace WTF

#endif // ENABLE(DFG_JIT)

