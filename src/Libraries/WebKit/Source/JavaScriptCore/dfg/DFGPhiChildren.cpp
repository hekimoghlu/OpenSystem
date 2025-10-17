/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#include "DFGPhiChildren.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "JSCJSValueInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace DFG {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PhiChildren);

PhiChildren::PhiChildren() = default;

PhiChildren::PhiChildren(Graph& graph)
{
    for (BasicBlock* block : graph.blocksInNaturalOrder()) {
        for (Node* node : *block) {
            if (node->op() != Upsilon)
                continue;
            
            m_children.add(node->phi(), List()).iterator->value.append(node);
        }
    }
}

PhiChildren::~PhiChildren() = default;

const PhiChildren::List& PhiChildren::upsilonsOf(Node* node) const
{
    ASSERT(node->op() == Phi);
    return m_children.find(node)->value;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

