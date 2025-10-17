/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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
#include "BytecodeLivenessAnalysis.h"

#include "BytecodeLivenessAnalysisInlines.h"
#include "BytecodeUseDef.h"
#include "CodeBlock.h"
#include "FullBytecodeLiveness.h"
#include "JSCJSValueInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BytecodeLivenessAnalysis);
WTF_MAKE_TZONE_ALLOCATED_IMPL(FullBytecodeLiveness);

BytecodeLivenessAnalysis::BytecodeLivenessAnalysis(CodeBlock* codeBlock)
    : m_graph(codeBlock, codeBlock->instructions())
{
    runLivenessFixpoint(codeBlock, codeBlock->instructions(), m_graph);

    if (UNLIKELY(Options::dumpBytecodeLivenessResults()))
        dumpResults(codeBlock);
}

std::unique_ptr<FullBytecodeLiveness> BytecodeLivenessAnalysis::computeFullLiveness(CodeBlock* codeBlock)
{
    FastBitVector out;

    size_t size = codeBlock->instructions().size();
    auto result = makeUnique<FullBytecodeLiveness>(size);

    for (auto& block : m_graph.basicBlocksInReverseOrder()) {
        if (block.isEntryBlock() || block.isExitBlock())
            continue;

        out = block.out();

        auto use = [&] (unsigned bitIndex) {
            // This is the use functor, so we set the bit.
            out[bitIndex] = true;
        };

        auto def = [&] (unsigned bitIndex) {
            // This is the def functor, so we clear the bit.
            out[bitIndex] = false;
        };

        auto& instructions = codeBlock->instructions();
        unsigned cursor = block.totalLength();
        for (unsigned i = block.delta().size(); i--;) {
            cursor -= block.delta()[i];
            BytecodeIndex bytecodeIndex = BytecodeIndex(block.leaderOffset() + cursor);
            auto instruction = instructions.at(bytecodeIndex);
            for (Checkpoint checkpoint = instruction->numberOfCheckpoints(); checkpoint--;) {
                ASSERT(checkpoint < instruction->size());
                bytecodeIndex = bytecodeIndex.withCheckpoint(checkpoint);

                stepOverBytecodeIndexDef(codeBlock, instructions, m_graph, bytecodeIndex, def);
                stepOverBytecodeIndexUseInExceptionHandler(codeBlock, instructions, m_graph, bytecodeIndex, use);
                result->m_usesAfter[result->toIndex(bytecodeIndex)] = out; // AfterUse point.
                stepOverBytecodeIndexUse(codeBlock, instructions, m_graph, bytecodeIndex, use);
                result->m_usesBefore[result->toIndex(bytecodeIndex)] = out; // BeforeUse point.
            }
        }
    }

    return result;
}

void BytecodeLivenessAnalysis::dumpResults(CodeBlock* codeBlock)
{
    dataLog("\nDumping bytecode liveness for ", *codeBlock, ":\n");
    const auto& instructions = codeBlock->instructions();
    unsigned i = 0;

    unsigned numberOfBlocks = m_graph.size();
    Vector<FastBitVector> predecessors(numberOfBlocks);
    for (auto& block : m_graph)
        predecessors[block.index()].resize(numberOfBlocks);
    for (auto& block : m_graph) {
        for (unsigned successorIndex : block.successors()) {
            unsigned blockIndex = block.index();
            predecessors[successorIndex][blockIndex] = true;
        }
    }

    auto dumpBitVector = [] (FastBitVector& bits) {
        for (unsigned j = 0; j < bits.numBits(); j++) {
            if (bits[j])
                dataLogF(" %u", j);
        }
    };

    for (auto& block : m_graph) {
        dataLogF("\nBytecode basic block %u: %p (offset: %u, length: %u)\n", i++, &block, block.leaderOffset(), block.totalLength());

        dataLogF("Predecessors:");
        dumpBitVector(predecessors[block.index()]);
        dataLogF("\n");

        dataLogF("Successors:");
        FastBitVector successors;
        successors.resize(numberOfBlocks);
        for (unsigned successorIndex : block.successors())
            successors[successorIndex] = true;
        dumpBitVector(successors); // Dump in sorted order.
        dataLogF("\n");

        if (block.isEntryBlock()) {
            dataLogF("Entry block %p\n", &block);
            continue;
        }
        if (block.isExitBlock()) {
            dataLogF("Exit block: %p\n", &block);
            continue;
        }
        for (unsigned bytecodeOffset = block.leaderOffset(); bytecodeOffset < block.leaderOffset() + block.totalLength();) {
            const auto currentInstruction = instructions.at(bytecodeOffset);

            dataLogF("Live variables:");
            FastBitVector liveBefore = getLivenessInfoAtInstruction(codeBlock, BytecodeIndex(bytecodeOffset));
            dumpBitVector(liveBefore);
            dataLogF("\n");
            codeBlock->dumpBytecode(WTF::dataFile(), currentInstruction);

            bytecodeOffset += currentInstruction->size();
        }

        dataLogF("Live variables:");
        FastBitVector liveAfter = block.out();
        dumpBitVector(liveAfter);
        dataLogF("\n");
    }
}

template<typename EnumType1, typename EnumType2>
constexpr bool enumValuesEqualAsIntegral(EnumType1 v1, EnumType2 v2)
{
    using IntType1 = typename std::underlying_type<EnumType1>::type;
    using IntType2 = typename std::underlying_type<EnumType2>::type;
    if constexpr (sizeof(IntType1) > sizeof(IntType2))
        return static_cast<IntType1>(v1) == static_cast<IntType1>(v2);
    else
        return static_cast<IntType2>(v1) == static_cast<IntType2>(v2);
}

WTF::BitSet<maxNumCheckpointTmps> tmpLivenessForCheckpoint(const CodeBlock& codeBlock, BytecodeIndex bytecodeIndex)
{
    WTF::BitSet<maxNumCheckpointTmps> result;
    Checkpoint checkpoint = bytecodeIndex.checkpoint();

    if (!checkpoint)
        return result;

    switch (codeBlock.instructions().at(bytecodeIndex)->opcodeID()) {
    case op_call_varargs:
    case op_tail_call_varargs:
    case op_construct_varargs:
    case op_super_construct_varargs: {
        static_assert(enumValuesEqualAsIntegral(OpCallVarargs::makeCall, OpTailCallVarargs::makeCall) && enumValuesEqualAsIntegral(OpCallVarargs::argCountIncludingThis, OpTailCallVarargs::argCountIncludingThis));
        static_assert(enumValuesEqualAsIntegral(OpCallVarargs::makeCall, OpConstructVarargs::makeCall) && enumValuesEqualAsIntegral(OpCallVarargs::argCountIncludingThis, OpConstructVarargs::argCountIncludingThis));
        static_assert(enumValuesEqualAsIntegral(OpCallVarargs::makeCall, OpSuperConstructVarargs::makeCall) && enumValuesEqualAsIntegral(OpCallVarargs::argCountIncludingThis, OpSuperConstructVarargs::argCountIncludingThis));
        if (checkpoint == OpCallVarargs::makeCall)
            result.set(OpCallVarargs::argCountIncludingThis);
        return result;
    }
    case op_iterator_open: {
        return result;
    }
    case op_iterator_next: {
        result.set(OpIteratorNext::nextResult);
        return result;
    }
    case op_instanceof: {
        return result;
    }
    default:
        break;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace JSC
