/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
#include "DFGPromotedHeapLocation.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

void PromotedLocationDescriptor::dump(PrintStream& out) const
{
    out.print(m_kind, "(", m_info, ")");
}

Node* PromotedHeapLocation::createHint(Graph& graph, NodeOrigin origin, Node* value)
{
    return graph.addNode(
        SpecNone, PutHint, origin, OpInfo(descriptor().imm1()), OpInfo(descriptor().imm2()),
        base()->defaultEdge(), value->defaultEdge());
}

void PromotedHeapLocation::dump(PrintStream& out) const
{
    out.print(kind(), "(", m_base, ", ", info(), ")");
}

} } // namespace JSC::DFG

namespace WTF {

using namespace JSC::DFG;

void printInternal(PrintStream& out, PromotedLocationKind kind)
{
    switch (kind) {
    case InvalidPromotedLocationKind:
        out.print("InvalidPromotedLocationKind");
        return;
        
    case StructurePLoc:
        out.print("StructurePLoc");
        return;

    case ActivationSymbolTablePLoc:
        out.print("ActivationSymbolTablePLoc");
        return;
        
    case NamedPropertyPLoc:
        out.print("NamedPropertyPLoc");
        return;

    case IndexedPropertyPLoc:
        out.print("IndexedPropertyPLoc");
        return;
        
    case ArgumentPLoc:
        out.print("ArgumentPLoc");
        return;
        
    case ArgumentCountPLoc:
        out.print("ArgumentCountPLoc");
        return;
        
    case ArgumentsCalleePLoc:
        out.print("ArgumentsCalleePLoc");
        return;

    case FunctionExecutablePLoc:
        out.print("FunctionExecutablePLoc");
        return;

    case FunctionActivationPLoc:
        out.print("FunctionActivationPLoc");
        return;

    case ActivationScopePLoc:
        out.print("ActivationScopePLoc");
        return;

    case ClosureVarPLoc:
        out.print("ClosureVarPLoc");
        return;

    case PublicLengthPLoc:
        out.print("PublicLengthPLoc");
        return;

    case VectorLengthPLoc:
        out.print("VectorLengthPLoc");
        return;

    case SpreadPLoc:
        out.print("SpreadPLoc");
        return;

    case NewArrayWithSpreadArgumentPLoc:
        out.print("NewArrayWithSpreadArgumentPLoc");
        return;

    case NewArrayBufferPLoc:
        out.print("NewArrayBufferPLoc");
        return;

    case RegExpObjectRegExpPLoc:
        out.print("RegExpObjectRegExpPLoc");
        return;

    case RegExpObjectLastIndexPLoc:
        out.print("RegExpObjectLastIndexPLoc");
        return;

    case InternalFieldObjectPLoc:
        out.print("InternalFieldObjectPLoc");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF;

#endif // ENABLE(DFG_JIT)

