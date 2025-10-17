/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 12, 2022.
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
#include "DFGTierUpCheckInjectionPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGNaturalLoops.h"
#include "DFGPhase.h"
#include "FTLCapabilities.h"
#include "FunctionAllowlist.h"
#include "JSCJSValueInlines.h"
#include <wtf/NeverDestroyed.h>

namespace JSC { namespace DFG {

static FunctionAllowlist& ensureGlobalFTLAllowlist()
{
    static LazyNeverDestroyed<FunctionAllowlist> ftlAllowlist;
    static std::once_flag initializeAllowlistFlag;
    std::call_once(initializeAllowlistFlag, [] {
        const char* functionAllowlistFile = Options::ftlAllowlist();
        ftlAllowlist.construct(functionAllowlistFile);
    });
    return ftlAllowlist;
}

using NaturalLoop = CPSNaturalLoop;

class TierUpCheckInjectionPhase : public Phase {
public:
    TierUpCheckInjectionPhase(Graph& graph)
        : Phase(graph, "tier-up check injection"_s)
    {
    }
    
    bool run()
    {
        RELEASE_ASSERT(m_graph.m_plan.isDFG());

        if (!Options::useFTLJIT())
            return false;
        
        if (m_graph.m_profiledBlock->m_didFailFTLCompilation)
            return false;

        if (!Options::bytecodeRangeToFTLCompile().isInRange(m_graph.m_profiledBlock->instructionsSize()))
            return false;

        if (!ensureGlobalFTLAllowlist().contains(m_graph.m_profiledBlock))
            return false;
        
#if ENABLE(FTL_JIT)
        FTL::CapabilityLevel level = FTL::canCompile(m_graph);
        if (level == FTL::CannotCompile)
            return false;
        
        if (!Options::useOSREntryToFTL())
            level = FTL::CanCompile;
        
        m_graph.ensureCPSNaturalLoops();
        CPSNaturalLoops& naturalLoops = *m_graph.m_cpsNaturalLoops;
        UncheckedKeyHashMap<const NaturalLoop*, BytecodeIndex> naturalLoopToLoopHint = buildNaturalLoopToLoopHintMap(naturalLoops);

        UncheckedKeyHashMap<BytecodeIndex, LoopHintDescriptor> tierUpHierarchy;

        InsertionSet insertionSet(m_graph);
        for (BlockIndex blockIndex = m_graph.numBlocks(); blockIndex--;) {
            BasicBlock* block = m_graph.block(blockIndex);
            if (!block)
                continue;

            for (unsigned nodeIndex = 0; nodeIndex < block->size(); ++nodeIndex) {
                Node* node = block->at(nodeIndex);
                if (node->op() != LoopHint)
                    continue;

                NodeOrigin origin = node->origin;
                bool canOSREnter = canOSREnterAtLoopHint(level, block, nodeIndex);

                NodeType tierUpType = CheckTierUpAndOSREnter;
                if (!canOSREnter)
                    tierUpType = CheckTierUpInLoop;
                insertionSet.insertNode(nodeIndex + 1, SpecNone, tierUpType, origin);

                auto bytecodeIndex = origin.semantic.bytecodeIndex();
                if (canOSREnter)
                    m_graph.m_plan.tierUpAndOSREnterBytecodes().append(bytecodeIndex);

                if (const NaturalLoop* loop = naturalLoops.innerMostLoopOf(block)) {
                    LoopHintDescriptor descriptor;
                    descriptor.canOSREnter = canOSREnter;

                    const NaturalLoop* outerLoop = loop;
                    while ((outerLoop = naturalLoops.innerMostOuterLoop(*outerLoop))) {
                        auto it = naturalLoopToLoopHint.find(outerLoop);
                        if (it != naturalLoopToLoopHint.end())
                            descriptor.osrEntryCandidates.append(it->value);
                    }
                    tierUpHierarchy.add(bytecodeIndex, WTFMove(descriptor));
                }
                break;
            }

            NodeAndIndex terminal = block->findTerminal();
            if (terminal.node->isFunctionTerminal()) {
                insertionSet.insertNode(
                    terminal.index, SpecNone, CheckTierUpAtReturn, terminal.node->origin);
            }

            insertionSet.execute(block);
        }

        // Add all the candidates that can be OSR Entered.
        for (auto entry : tierUpHierarchy) {
            Vector<BytecodeIndex> tierUpCandidates;
            for (BytecodeIndex bytecodeIndex : entry.value.osrEntryCandidates) {
                auto descriptorIt = tierUpHierarchy.find(bytecodeIndex);
                if (descriptorIt != tierUpHierarchy.end()
                    && descriptorIt->value.canOSREnter)
                    tierUpCandidates.append(bytecodeIndex);
            }

            if (!tierUpCandidates.isEmpty())
                m_graph.m_plan.tierUpInLoopHierarchy().ensure(entry.key, [&] { return FixedVector<BytecodeIndex>(WTFMove(tierUpCandidates)); });
        }
        m_graph.m_plan.setWillTryToTierUp(true);
        return true;
#else // ENABLE(FTL_JIT)
        RELEASE_ASSERT_NOT_REACHED();
        return false;
#endif // ENABLE(FTL_JIT)
    }

private:
#if ENABLE(FTL_JIT)
    struct LoopHintDescriptor {
        Vector<BytecodeIndex> osrEntryCandidates;
        bool canOSREnter;
    };

    bool canOSREnterAtLoopHint(FTL::CapabilityLevel level, const BasicBlock* block, unsigned nodeIndex)
    {
        Node* node = block->at(nodeIndex);
        ASSERT(node->op() == LoopHint);

        NodeOrigin origin = node->origin;
        if (level != FTL::CanCompileAndOSREnter || origin.semantic.inlineCallFrame())
            return false;

        // We only put OSR checks for the first LoopHint in the block. Note that
        // more than one LoopHint could happen in cases where we did a lot of CFG
        // simplification in the bytecode parser, but it should be very rare.
        for (unsigned subNodeIndex = nodeIndex; subNodeIndex--;) {
            if (!block->at(subNodeIndex)->isSemanticallySkippable())
                return false;
        }
        return true;
    }

    UncheckedKeyHashMap<const NaturalLoop*, BytecodeIndex> buildNaturalLoopToLoopHintMap(const CPSNaturalLoops& naturalLoops)
    {
        UncheckedKeyHashMap<const NaturalLoop*, BytecodeIndex> naturalLoopsToLoopHint;

        for (BasicBlock* block : m_graph.blocksInNaturalOrder()) {
            for (unsigned nodeIndex = 0; nodeIndex < block->size(); ++nodeIndex) {
                Node* node = block->at(nodeIndex);
                if (node->op() != LoopHint)
                    continue;

                if (const NaturalLoop* loop = naturalLoops.innerMostLoopOf(block)) {
                    BytecodeIndex bytecodeIndex = node->origin.semantic.bytecodeIndex();
                    naturalLoopsToLoopHint.add(loop, bytecodeIndex);
                }
                break;
            }
        }
        return naturalLoopsToLoopHint;
    }
#endif
};

bool performTierUpCheckInjection(Graph& graph)
{
    return runPhase<TierUpCheckInjectionPhase>(graph);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)


