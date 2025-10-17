/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 20, 2025.
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
#include "DFGValueSource.h"

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

void ValueSource::dump(PrintStream& out) const
{
    switch (kind()) {
    case SourceNotSet:
        out.print("NotSet");
        break;
    case SourceIsDead:
        out.print("IsDead");
        break;
    case ValueInJSStack:
        out.print("JS:", virtualRegister());
        break;
    case Int32InJSStack:
        out.print("Int32:", virtualRegister());
        break;
    case Int52InJSStack:
        out.print("Int52:", virtualRegister());
        break;
    case CellInJSStack:
        out.print("Cell:", virtualRegister());
        break;
    case BooleanInJSStack:
        out.print("Bool:", virtualRegister());
        break;
    case DoubleInJSStack:
        out.print("Double:", virtualRegister());
        break;
    case HaveNode:
        out.print("Node(", m_value, ")");
        break;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        break;
    }
}

void ValueSource::dumpInContext(PrintStream& out, DumpContext*) const
{
    dump(out);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

