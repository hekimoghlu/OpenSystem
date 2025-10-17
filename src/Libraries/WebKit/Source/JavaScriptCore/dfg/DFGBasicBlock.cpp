/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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
#include "DFGBasicBlock.h"

#if ENABLE(DFG_JIT)

#include "JSCJSValueInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC { namespace DFG {

using BasicBlockSSAData = BasicBlock::SSAData;

WTF_MAKE_TZONE_ALLOCATED_IMPL(BasicBlockSSAData);

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(BasicBlock);

BasicBlock::BasicBlock(BytecodeIndex bytecodeBegin, unsigned numArguments, unsigned numLocals, unsigned numTmps, float executionCount)
    : bytecodeBegin(bytecodeBegin)
    , index(NoBlock)
    , cfaStructureClobberStateAtHead(StructuresAreWatched)
    , cfaStructureClobberStateAtTail(StructuresAreWatched)
    , cfaBranchDirection(InvalidBranchDirection)
    , cfaHasVisited(false)
    , cfaShouldRevisit(false)
    , cfaDidFinish(true)
    , intersectionOfCFAHasVisited(true)
    , isOSRTarget(false)
    , isCatchEntrypoint(false)
#if ASSERT_ENABLED
    , isLinked(false)
#endif
    , isReachable(false)
    , variablesAtHead(numArguments, numLocals, numTmps)
    , variablesAtTail(numArguments, numLocals, numTmps)
    , valuesAtHead(numArguments, numLocals, numTmps)
    , valuesAtTail(numArguments, numLocals, numTmps)
    , intersectionOfPastValuesAtHead(numArguments, numLocals, numTmps, AbstractValue::fullTop())
    , executionCount(executionCount)
{
}

BasicBlock::~BasicBlock() = default;

void BasicBlock::ensureLocals(unsigned newNumLocals)
{
    variablesAtHead.ensureLocals(newNumLocals);
    variablesAtTail.ensureLocals(newNumLocals);
    valuesAtHead.ensureLocals(newNumLocals);
    valuesAtTail.ensureLocals(newNumLocals);
    intersectionOfPastValuesAtHead.ensureLocals(newNumLocals, AbstractValue::fullTop());
}

void BasicBlock::ensureTmps(unsigned newNumTmps)
{
    variablesAtHead.ensureTmps(newNumTmps);
    variablesAtTail.ensureTmps(newNumTmps);
    valuesAtHead.ensureTmps(newNumTmps);
    valuesAtTail.ensureTmps(newNumTmps);
    intersectionOfPastValuesAtHead.ensureTmps(newNumTmps, AbstractValue::fullTop());
}

void BasicBlock::replaceTerminal(Graph& graph, Node* node)
{
    NodeAndIndex result = findTerminal();
    if (!result)
        append(node);
    else {
        m_nodes.insert(result.index + 1, node);
        result.node->remove(graph);
    }
    
    ASSERT(terminal());
}

bool BasicBlock::isInPhis(Node* node) const
{
    for (size_t i = 0; i < phis.size(); ++i) {
        if (phis[i] == node)
            return true;
    }
    return false;
}

bool BasicBlock::isInBlock(Node* myNode) const
{
    for (size_t i = 0; i < numNodes(); ++i) {
        if (node(i) == myNode)
            return true;
    }
    return false;
}

void BasicBlock::removePredecessor(BasicBlock* block)
{
    for (unsigned i = 0; i < predecessors.size(); ++i) {
        if (predecessors[i] != block)
            continue;
        predecessors[i] = predecessors.last();
        predecessors.removeLast();
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void BasicBlock::replacePredecessor(BasicBlock* from, BasicBlock* to)
{
    for (unsigned i = predecessors.size(); i--;) {
        if (predecessors[i] != from)
            continue;
        predecessors[i] = to;
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

void BasicBlock::dump(PrintStream& out) const
{
    out.print("#", index);
}

BasicBlock::SSAData::SSAData(BasicBlock* block)
{
    availabilityAtHead.m_locals = Operands<Availability>(OperandsLike, block->variablesAtHead);
    availabilityAtTail.m_locals = Operands<Availability>(OperandsLike, block->variablesAtHead);
}

BasicBlock::SSAData::~SSAData() { }

} } // namespace JSC::DFG

namespace WTF {

void printInternal(PrintStream& out, JSC::DFG::BasicBlock* block)
{
    out.print(*block);
}

}

#endif // ENABLE(DFG_JIT)

