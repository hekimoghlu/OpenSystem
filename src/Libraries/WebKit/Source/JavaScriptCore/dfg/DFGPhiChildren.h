/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 2, 2024.
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
#pragma once

#if ENABLE(DFG_JIT)

#include "DFGNode.h"
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace JSC { namespace DFG {

class Graph;

class PhiChildren {
    WTF_MAKE_TZONE_ALLOCATED(PhiChildren);
public:
    typedef Vector<Node*, 3> List;
    
    PhiChildren();
    PhiChildren(Graph&);
    ~PhiChildren();
    
    // The list of Upsilons that point to the children of the Phi.
    const List& upsilonsOf(Node*) const;
    
    template<typename Functor>
    void forAllIncomingValues(Node* node, const Functor& functor)
    {
        for (Node* upsilon : upsilonsOf(node))
            functor(upsilon->child1().node());
    }
    
    // This walks the Phi graph.
    template<typename Functor>
    void forAllTransitiveIncomingValues(Node* node, const Functor& functor)
    {
        if (node->op() != Phi) {
            functor(node);
            return;
        }
        UncheckedKeyHashSet<Node*> seen;
        Vector<Node*> worklist;
        seen.add(node);
        worklist.append(node);
        while (!worklist.isEmpty()) {
            Node* currentNode = worklist.takeLast();
            forAllIncomingValues(
                currentNode,
                [&] (Node* incomingNode) {
                    if (incomingNode->op() == Phi) {
                        if (seen.add(incomingNode).isNewEntry)
                            worklist.append(incomingNode);
                    } else
                        functor(incomingNode);
                });
        }
    }
    
private:
    UncheckedKeyHashMap<Node*, List> m_children;
};

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
