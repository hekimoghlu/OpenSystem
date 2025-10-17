/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 24, 2022.
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
#include "DFGInsertionSet.h"

#if ENABLE(DFG_JIT)

#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

void InsertionSet::insertSlow(const Insertion& insertion)
{
    ASSERT(!m_insertions.isEmpty());
    ASSERT(m_insertions.last().index() > insertion.index());
    
    for (size_t index = m_insertions.size() - 1; index--;) {
        if (m_insertions[index].index() <= insertion.index()) {
            m_insertions.insert(index + 1, insertion);
            return;
        }
    }

    m_insertions.insert(0, insertion);
}

size_t InsertionSet::execute(BasicBlock* block)
{
    return executeInsertions(*block, m_insertions);
}

Node* InsertionSet::insertConstant(size_t index, NodeOrigin origin, FrozenValue* value, NodeType op)
{
    return insertNode(index, speculationFromValue(value->value()), op, origin, OpInfo(value));
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

