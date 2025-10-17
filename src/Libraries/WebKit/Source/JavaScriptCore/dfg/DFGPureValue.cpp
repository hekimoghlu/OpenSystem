/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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
#include "DFGPureValue.h"

#if ENABLE(DFG_JIT)

#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

void PureValue::dump(PrintStream& out) const
{
    out.print(Graph::opName(op()));
    out.print("("_s);
    CommaPrinter comma;
    if (isVarargs()) {
        for (unsigned i = 0; i < m_children.numChildren(); ++i)
            out.print(comma, m_graph->m_varArgChildren[m_children.firstChild() + i].sanitized());
    } else {
        for (unsigned i = 0; i < AdjacencyList::Size; ++i) {
            if (m_children.child(i))
                out.print(comma, m_children.child(i));
        }
    }
    if (m_info)
        out.print(comma, m_info);
    out.print(")"_s);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

