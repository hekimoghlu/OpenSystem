/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 26, 2022.
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
#include "DFGMovHintRemovalPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGEpoch.h"
#include "DFGForAllKills.h"
#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGMayExit.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"
#include "OperandsInlines.h"

namespace JSC { namespace DFG {

namespace {

namespace DFGMovHintRemovalPhaseInternal {
static constexpr bool verbose = false;
}

class MovHintRemovalPhase : public Phase {
public:
    MovHintRemovalPhase(Graph& graph)
        : Phase(graph, "MovHint removal"_s)
        , m_insertionSet(graph)
        , m_state(OperandsLike, graph.block(0)->variablesAtHead)
        , m_changed(false)
    {
    }

    bool run()
    {
        if (DFGMovHintRemovalPhaseInternal::verbose) {
            dataLog("Graph before MovHint removal:\n");
            m_graph.dump();
        }

        for (BasicBlock* block : m_graph.blocksInNaturalOrder())
            handleBlock(block);

        m_insertionSet.execute(m_graph.block(0));

        return m_changed;
    }

private:
    void handleBlock(BasicBlock* block)
    {
        if (DFGMovHintRemovalPhaseInternal::verbose)
            dataLog("Handing block ", pointerDump(block), "\n");

        // A MovHint is unnecessary if the local dies before it is used. We answer this question by
        // maintaining the current exit epoch, and associating an epoch with each local. When a
        // local dies, it gets the current exit epoch. If a MovHint occurs in the same epoch as its
        // local, then it means there was no exit between the local's death and the MovHint - i.e.
        // the MovHint is unnecessary.

        Epoch currentEpoch = Epoch::first();

        m_state.fill(Epoch());
        m_graph.forAllLiveInBytecode(
            block->terminal()->origin.forExit,
            [&] (Operand reg) {
                m_state.operand(reg) = currentEpoch;
            });

        if (DFGMovHintRemovalPhaseInternal::verbose)
            dataLog("    Locals at ", block->terminal()->origin.forExit, ": ", m_state, "\n");

        // Assume that blocks after we exit.
        currentEpoch.bump();

        for (unsigned nodeIndex = block->size(); nodeIndex--;) {
            Node* node = block->at(nodeIndex);

            if (node->op() == MovHint) {
                Epoch localEpoch = m_state.operand(node->unlinkedOperand());
                if (DFGMovHintRemovalPhaseInternal::verbose)
                    dataLog("    At ", node, " (", node->unlinkedOperand(), "): current = ", currentEpoch, ", local = ", localEpoch, "\n");
                if (!localEpoch || localEpoch == currentEpoch) {
                    // Now, MovHint will put bottom value to dead locals. This means that if you insert a new DFG node which introduce
                    // a new OSR exit, then it gets confused with the already-determined-dead locals. So this phase runs at very end of
                    // DFG pipeline, and we do not insert a node having a new OSR exit (if it is existing OSR exit, or if it does not exit,
                    // then it is totally fine).
                    node->setOpAndDefaultFlags(ZombieHint);
                    UseKind useKind = node->child1().useKind();
                    Node* constant = m_constants.ensure(static_cast<std::underlying_type_t<UseKind>>(useKind), [&]() -> Node* {
                        return m_insertionSet.insertBottomConstantForUse(0, m_graph.block(0)->at(0)->origin, useKind).node();
                    }).iterator->value;
                    node->child1() = Edge(constant, useKind);
                    m_changed = true;
                }
                m_state.operand(node->unlinkedOperand()) = Epoch();
            }

            if (mayExit(m_graph, node) != DoesNotExit)
                currentEpoch.bump();

            if (nodeIndex) {
                forAllKilledOperands(
                    m_graph, block->at(nodeIndex - 1), node,
                    [&] (Operand operand) {
                        // This function is a bit sloppy - it might claim to kill a local even if
                        // it's still live after. We need to protect against that.
                        if (!!m_state.operand(operand))
                            return;

                        if (DFGMovHintRemovalPhaseInternal::verbose)
                            dataLog("    Killed operand at ", node, ": ", operand, "\n");
                        m_state.operand(operand) = currentEpoch;
                    });
            }
        }
    }

    InsertionSet m_insertionSet;
    UncheckedKeyHashMap<std::underlying_type_t<UseKind>, Node*, WTF::IntHash<std::underlying_type_t<UseKind>>, WTF::UnsignedWithZeroKeyHashTraits<std::underlying_type_t<UseKind>>> m_constants;
    Operands<Epoch> m_state;
    bool m_changed;
};

} // anonymous namespace

bool performMovHintRemoval(Graph& graph)
{
    return runPhase<MovHintRemovalPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

