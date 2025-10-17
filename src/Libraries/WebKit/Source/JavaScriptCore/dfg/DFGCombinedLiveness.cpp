/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 8, 2022.
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
#include "DFGCombinedLiveness.h"

#if ENABLE(DFG_JIT)

#include "DFGAvailabilityMap.h"
#include "DFGBlockMapInlines.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

static void addBytecodeLiveness(Graph& graph, AvailabilityMap& availabilityMap, NodeSet& seen, Node* node)
{
    graph.forAllLiveInBytecode(
        node->origin.forExit,
        [&] (Operand reg) {
            availabilityMap.closeStartingWithLocal(
                reg,
                [&] (Node* node) -> bool {
                    return seen.contains(node);
                },
                [&] (Node* node) -> bool {
                    return seen.add(node).isNewEntry;
                });
        });
}

NodeSet liveNodesAtHead(Graph& graph, BasicBlock* block)
{
    NodeSet seen;
    for (NodeFlowProjection node : block->ssa->liveAtHead) {
        if (node.kind() == NodeFlowProjection::Primary)
            seen.addVoid(node.node());
    }

    addBytecodeLiveness(graph, block->ssa->availabilityAtHead, seen, block->at(0));
    return seen;
}

CombinedLiveness::CombinedLiveness(Graph& graph)
    : liveAtHead(graph)
    , liveAtTail(graph)
{
    // First compute 
    // - The liveAtHead for each block.
    // - The liveAtTail for blocks that won't properly propagate
    //   the information based on their empty successor list.
    for (BasicBlock* block : graph.blocksInNaturalOrder()) {
        liveAtHead[block] = liveNodesAtHead(graph, block);

        // If we don't have successors, we can't rely on the propagation below. This doesn't usually
        // do anything for terminal blocks, since the last node is usually a return, so nothing is live
        // after it. However, we may also have the end of the basic block be:
        //
        // ForceOSRExit
        // Unreachable
        //
        // And things may definitely be live in bytecode at that point in the program.
        if (!block->numSuccessors()) {
            NodeSet seen;
            addBytecodeLiveness(graph, block->ssa->availabilityAtTail, seen, block->last());
            liveAtTail[block] = seen;
        }
    }
    
    // Now compute the liveAtTail by unifying the liveAtHead of the successors.
    for (BasicBlock* block : graph.blocksInNaturalOrder()) {
        for (BasicBlock* successor : block->successors()) {
            for (Node* node : liveAtHead[successor])
                liveAtTail[block].addVoid(node);
        }
    }
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

