/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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
#include "DOMJITAbstractHeap.h"

#if ENABLE(JIT)

namespace JSC { namespace DOMJIT {

void AbstractHeap::compute(unsigned begin)
{
    unsigned current = begin;
    // Increment the end of the range.
    if (m_children.isEmpty()) {
        m_range = HeapRange(begin, current + 1);
        return;
    }
    for (auto& child : m_children) {
        child->compute(current);
        current = child->range().end();
    }
    ASSERT(begin < UINT16_MAX);
    ASSERT(current <= UINT16_MAX);
    m_range = HeapRange(begin, current);
}

void AbstractHeap::dump(PrintStream& out) const
{
    shallowDump(out);
    if (m_parent)
        out.print("->", *m_parent);
}

void AbstractHeap::shallowDump(PrintStream& out) const
{
    out.print(m_name, "<", m_range, ">");
}

void AbstractHeap::deepDump(PrintStream& out, unsigned indent) const
{
    auto printIndent = [&] () {
        for (unsigned i = indent; i--;)
            out.print("    ");
    };

    printIndent();
    shallowDump(out);

    if (m_children.isEmpty()) {
        out.print("\n");
        return;
    }

    out.print(":\n");
    for (auto* child : m_children)
        child->deepDump(out, indent + 1);
}

} }

#endif
