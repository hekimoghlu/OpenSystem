/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 24, 2023.
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
#include "DFGLivenessAnalysisPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGBlockMapInlines.h"
#include "DFGFlowIndexing.h"
#include "DFGGraph.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"
#include <wtf/BitVector.h>
#include <wtf/IndexSparseSet.h>

namespace JSC { namespace DFG {

namespace {

// Uncomment this to log hashtable operations.
// static const char templateString[] = "unsigned, DefaultHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>";
// typedef LoggingHashSet<templateString, unsigned, DefaultHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> LiveSet;

typedef UncheckedKeyHashSet<unsigned, DefaultHash<unsigned>, WTF::UnsignedWithZeroKeyHashTraits<unsigned>> LiveSet;

typedef IndexSparseSet<unsigned, DefaultIndexSparseSetTraits<unsigned>, UnsafeVectorOverflow> Workset;

class LivenessAnalysisPhase : public Phase {
public:
    LivenessAnalysisPhase(Graph& graph)
        : Phase(graph, "liveness analysis"_s)
        , m_dirtyBlocks(m_graph.numBlocks())
        , m_indexing(*m_graph.m_indexingCache)
        , m_liveAtHead(m_graph)
        , m_liveAtTail(m_graph)
    {
        m_graph.m_indexingCache->recompute();
        m_workset = makeUnique<Workset>(m_graph.m_indexingCache->numIndices());
    }

    bool run()
    {
        // Start with all valid block dirty.
        BlockIndex numBlock = m_graph.numBlocks();
        m_dirtyBlocks.ensureSize(numBlock);
        for (BlockIndex blockIndex = 0; blockIndex < numBlock; ++blockIndex) {
            if (m_graph.block(blockIndex))
                m_dirtyBlocks.quickSet(blockIndex);
        }

        // Fixpoint until we do not add any new live values at tail.
        bool changed;
        do {
            changed = false;

            for (BlockIndex blockIndex = numBlock; blockIndex--;) {
                if (!m_dirtyBlocks.quickClear(blockIndex))
                    continue;

                changed |= processBlock(blockIndex);
            }
        } while (changed);

        // Update the per-block node list for live and tail.
        for (BlockIndex blockIndex = numBlock; blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;

            {
                block->ssa->liveAtHead = m_liveAtHead[blockIndex].map([this](auto index) {
                    return m_indexing.nodeProjection(index);
                });
            }
            {
                block->ssa->liveAtTail = WTF::map(m_liveAtTail[blockIndex], [this](auto index) {
                    return m_indexing.nodeProjection(index);
                });
            }
        }

        return true;
    }

private:
    bool processBlock(BlockIndex blockIndex)
    {
        BasicBlock* block = m_graph.block(blockIndex);
        ASSERT_WITH_MESSAGE(block, "Only dirty blocks needs updates. A null block should never be dirty.");

        m_workset->clear();
        for (unsigned index : m_liveAtTail[blockIndex])
            m_workset->add(index);

        for (unsigned nodeIndex = block->size(); nodeIndex--;) {
            Node* node = block->at(nodeIndex);

            auto handleEdge = [&] (Edge& edge) {
                bool newEntry = m_workset->add(m_indexing.index(edge.node()));
                edge.setKillStatus(newEntry ? DoesKill : DoesNotKill);
            };
            
            switch (node->op()) {
            case Upsilon: {
                ASSERT_WITH_MESSAGE(!m_workset->contains(node->index()), "Upsilon should not be used as defs by other nodes.");

                Node* phi = node->phi();
                m_workset->remove(m_indexing.shadowIndex(phi));
                handleEdge(node->child1());
                break;
            }
            case Phi: {
                m_workset->remove(m_indexing.index(node));
                m_workset->add(m_indexing.shadowIndex(node));
                break;
            }
            default:
                m_workset->remove(m_indexing.index(node));
                m_graph.doToChildren(node, handleEdge);
                break;
            }
        }

        // Update live at head.
        Vector<unsigned, 0, UnsafeVectorOverflow, 1>& liveAtHead = m_liveAtHead[blockIndex];
        if (m_workset->size() == liveAtHead.size())
            return false;

        for (unsigned liveIndexAtHead : liveAtHead)
            m_workset->remove(liveIndexAtHead);
        ASSERT(!m_workset->isEmpty());

        liveAtHead.appendRange(m_workset->begin(), m_workset->end());

        bool changedPredecessor = false;
        for (BasicBlock* predecessor : block->predecessors) {
            LiveSet& liveAtTail = m_liveAtTail[predecessor];
            for (unsigned newValue : *m_workset) {
                if (liveAtTail.add(newValue)) {
                    if (!m_dirtyBlocks.quickSet(predecessor->index))
                        changedPredecessor = true;
                }
            }
        }
        return changedPredecessor;
    }

    // Blocks with new live values at tail.
    BitVector m_dirtyBlocks;
    
    FlowIndexing& m_indexing;

    // Live values per block edge.
    BlockMap<Vector<unsigned, 0, UnsafeVectorOverflow, 1>> m_liveAtHead;
    BlockMap<LiveSet> m_liveAtTail;

    // Single sparse set allocated once and used by every basic block.
    std::unique_ptr<Workset> m_workset;
};

} // anonymous namespace

bool performGraphPackingAndLivenessAnalysis(Graph& graph)
{
    graph.packNodeIndices();
#ifndef NDEBUG
    graph.clearAbstractValues();
#endif
    return runPhase<LivenessAnalysisPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

