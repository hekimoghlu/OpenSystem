/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#include "DFGLiveCatchVariablePreservationPhase.h"

#if ENABLE(DFG_JIT)

#include "DFGBlockSet.h"
#include "DFGGraph.h"
#include "DFGInsertionSet.h"
#include "DFGPhase.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

class LiveCatchVariablePreservationPhase {
public:
    LiveCatchVariablePreservationPhase(Graph& graph)
        : m_graph(graph)
    {
    }

    bool run()
    {
        DFG_ASSERT(m_graph, nullptr, m_graph.m_form == LoadStore);

        if (!m_graph.m_hasExceptionHandlers)
            return false;

        m_graph.determineReachability();

        InsertionSet insertionSet(m_graph);
        for (BasicBlock* block : m_graph.blocksInNaturalOrder()) {
            if (block->isReachable) {
                handleBlockForTryCatch(block, insertionSet);
                insertionSet.execute(block);
            }
        }

        m_graph.clearReachability();
        return true;
    }

    bool isValidFlushLocation(BasicBlock* startingBlock, unsigned index, Operand operand)
    {
        // This code is not meant to be fast. We just use it for assertions. If we got liveness wrong,
        // this function would return false for a Flush that we insert.
        Vector<BasicBlock*, 4> worklist;
        BlockSet seen;

        auto addPredecessors = [&] (BasicBlock* block) {
            for (BasicBlock* predecessor : block->predecessors) {
                bool isNewEntry = seen.add(predecessor);
                if (isNewEntry)
                    worklist.append(predecessor);
            }
        };

        auto flushIsDefinitelyInvalid = [&] (BasicBlock* block, unsigned index) {
            bool allGood = false;
            for (unsigned i = index; i--; ) {
                if (block->at(i)->accessesStack(m_graph) && block->at(i)->operand() == operand) {
                    allGood = true;
                    break;
                }
            }

            if (allGood)
                return false;

            if (block->predecessors.isEmpty()) {
                // This is a root block. We proved we reached here, therefore we can't Flush, as
                // it'll make this local live at the start of a root block, which is invalid IR.
                return true;
            }

            addPredecessors(block);
            return false;
        };

        if (flushIsDefinitelyInvalid(startingBlock, index))
            return false;

        while (!worklist.isEmpty()) {
            BasicBlock* block = worklist.takeLast();
            if (flushIsDefinitelyInvalid(block, block->size()))
                return false;
        }
        return true;
    }


    void handleBlockForTryCatch(BasicBlock* block, InsertionSet& insertionSet)
    {
        HandlerInfo* currentExceptionHandler = nullptr;
        Operands<bool> liveAtCatchHead(0, m_graph.block(0)->variablesAtTail.numberOfLocals(), m_graph.block(0)->variablesAtTail.numberOfTmps());

        HandlerInfo* cachedHandlerResult;
        CodeOrigin cachedCodeOrigin;
        auto catchHandler = [&] (CodeOrigin origin) -> HandlerInfo* {
            ASSERT(origin);
            if (origin == cachedCodeOrigin)
                return cachedHandlerResult;

            BytecodeIndex bytecodeIndexToCheck = origin.bytecodeIndex();

            cachedCodeOrigin = origin;

            while (1) {
                InlineCallFrame* inlineCallFrame = origin.inlineCallFrame();
                CodeBlock* codeBlock = m_graph.baselineCodeBlockFor(inlineCallFrame);
                if (HandlerInfo* handler = codeBlock->handlerForBytecodeIndex(bytecodeIndexToCheck)) {
                    liveAtCatchHead.fill(false);

                    BytecodeIndex catchBytecodeIndex = BytecodeIndex(handler->target);
                    m_graph.forAllLocalsAndTmpsLiveInBytecode(CodeOrigin(catchBytecodeIndex, inlineCallFrame), [&] (Operand operand) {
                        liveAtCatchHead.operand(operand) = true;
                    });

                    cachedHandlerResult = handler;
                    break;
                }

                if (!inlineCallFrame) {
                    cachedHandlerResult = nullptr;
                    break;
                }

                bytecodeIndexToCheck = inlineCallFrame->directCaller.bytecodeIndex();
                origin = inlineCallFrame->directCaller;
            }

            return cachedHandlerResult;
        };

        Operands<VariableAccessData*> currentBlockAccessData(OperandsLike, block->variablesAtTail, nullptr);

        auto flushEverything = [&] (NodeOrigin origin, unsigned index) {
            RELEASE_ASSERT(currentExceptionHandler);
            auto flush = [&] (Operand operand) {
                if (operand.isArgument() || liveAtCatchHead.operand(operand)) {

                    ASSERT(isValidFlushLocation(block, index, operand));

                    VariableAccessData* accessData = currentBlockAccessData.operand(operand);
                    if (!accessData)
                        accessData = newVariableAccessData(operand);

                    currentBlockAccessData.operand(operand) = accessData;

                    insertionSet.insertNode(index, SpecNone, 
                        Flush, origin, OpInfo(accessData));
                }
            };

            for (unsigned local = 0; local < block->variablesAtTail.numberOfLocals(); local++)
                flush(virtualRegisterForLocal(local));
            for (unsigned tmp = 0; tmp < block->variablesAtTail.numberOfTmps(); ++tmp)
                flush(Operand::tmp(tmp));
            flush(VirtualRegister(CallFrame::thisArgumentOffset()));
        };

        for (unsigned nodeIndex = 0; nodeIndex < block->size(); nodeIndex++) {
            Node* node = block->at(nodeIndex);

            {
                HandlerInfo* newHandler = catchHandler(node->origin.semantic);
                if (newHandler != currentExceptionHandler && currentExceptionHandler)
                    flushEverything(node->origin, nodeIndex);
                currentExceptionHandler = newHandler;
            }

            if (currentExceptionHandler && (node->op() == SetLocal || node->op() == SetArgumentDefinitely || node->op() == SetArgumentMaybe)) {
                Operand operand = node->operand();
                if (operand.isArgument() || liveAtCatchHead.operand(operand)) {
                    ASSERT(isValidFlushLocation(block, nodeIndex, operand));

                    VariableAccessData* variableAccessData = currentBlockAccessData.operand(operand);
                    if (!variableAccessData)
                        variableAccessData = newVariableAccessData(operand);

                    insertionSet.insertNode(nodeIndex, SpecNone, 
                        Flush, node->origin, OpInfo(variableAccessData));
                }
            }

            if (node->accessesStack(m_graph))
                currentBlockAccessData.operand(node->operand()) = node->variableAccessData();
        }

        if (currentExceptionHandler) {
            NodeOrigin origin = block->at(block->size() - 1)->origin;
            flushEverything(origin, block->size());
        }
    }

    VariableAccessData* newVariableAccessData(Operand operand)
    {
        ASSERT(!operand.isConstant());

        return &m_graph.m_variableAccessData.alloc(operand);
    }

    Graph& m_graph;
};

bool performLiveCatchVariablePreservationPhase(Graph& graph)
{
    return LiveCatchVariablePreservationPhase(graph).run();
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
