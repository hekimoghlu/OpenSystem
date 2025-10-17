/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#include "DFGVariableAccessDataDump.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGVariableAccessData.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

VariableAccessDataDump::VariableAccessDataDump(Graph& graph, VariableAccessData* data)
    : m_graph(graph)
    , m_data(data)
{
}

void VariableAccessDataDump::dump(PrintStream& out) const
{
    unsigned index = std::numeric_limits<unsigned>::max();
    for (unsigned i = 0; i < m_graph.m_variableAccessData.size(); ++i) {
        if (&m_graph.m_variableAccessData[i] == m_data) {
            index = i;
            break;
        }
    }
    
    ASSERT(index != std::numeric_limits<unsigned>::max());
    
    do {
        out.print(CharacterDump('A' + (index % 26)));
        index /= 26;
    } while (index);
    
    if (m_data->shouldNeverUnbox())
        out.print("!");
    else if (!m_data->shouldUnboxIfPossible())
        out.print("~");

    out.print(AbbreviatedSpeculationDump(m_data->prediction()), "/", m_data->flushFormat());
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)


