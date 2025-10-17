/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 6, 2024.
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
#include "ControlFlowProfiler.h"

#include "VM.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ControlFlowProfiler);

ControlFlowProfiler::ControlFlowProfiler()
    : m_dummyBasicBlock(BasicBlockLocation(-1, -1))
{
}

ControlFlowProfiler::~ControlFlowProfiler()
{
    for (const BlockLocationCache& cache : m_sourceIDBuckets.values()) {
        for (BasicBlockLocation* block : cache.values())
            delete block;
    }
}

BasicBlockLocation* ControlFlowProfiler::getBasicBlockLocation(SourceID sourceID, int startOffset, int endOffset)
{
    auto addResult = m_sourceIDBuckets.add(sourceID, BlockLocationCache());
    BlockLocationCache& blockLocationCache = addResult.iterator->value;
    BasicBlockKey key(startOffset, endOffset);
    auto addResultForBasicBlock = blockLocationCache.add(key, nullptr);
    if (addResultForBasicBlock.isNewEntry)
        addResultForBasicBlock.iterator->value = new BasicBlockLocation(startOffset, endOffset);
    return addResultForBasicBlock.iterator->value;
}

void ControlFlowProfiler::dumpData() const
{
    auto iter = m_sourceIDBuckets.begin();
    auto end = m_sourceIDBuckets.end();
    for (; iter != end; ++iter) {
        dataLog("SourceID: ", iter->key, "\n");
        for (const BasicBlockLocation* block : iter->value.values())
            block->dumpData();
    }
}

Vector<BasicBlockRange> ControlFlowProfiler::getBasicBlocksForSourceID(SourceID sourceID, VM& vm) const
{
    Vector<BasicBlockRange> result(0);
    auto bucketFindResult = m_sourceIDBuckets.find(sourceID);
    if (bucketFindResult == m_sourceIDBuckets.end())
        return result;

    const BlockLocationCache& cache = bucketFindResult->value;
    for (const BasicBlockLocation* block : cache.values()) {
        bool hasExecuted = block->hasExecuted();
        size_t executionCount = block->executionCount();
        const Vector<BasicBlockLocation::Gap>& blockRanges = block->getExecutedRanges();
        for (BasicBlockLocation::Gap gap : blockRanges) {
            BasicBlockRange range;
            range.m_hasExecuted = hasExecuted;
            range.m_executionCount = executionCount;
            range.m_startOffset = gap.first;
            range.m_endOffset = gap.second;
            result.append(range);
        }
    }

    const Vector<std::tuple<bool, unsigned, unsigned>>& functionRanges = vm.functionHasExecutedCache()->getFunctionRanges(sourceID);
    for (const auto& functionRange : functionRanges) {
        BasicBlockRange range;
        range.m_hasExecuted = std::get<0>(functionRange);
        range.m_startOffset = static_cast<int>(std::get<1>(functionRange));
        range.m_endOffset = static_cast<int>(std::get<2>(functionRange));
        range.m_executionCount = range.m_hasExecuted ? 1 : 0; // This is a hack. We don't actually count this.
        result.append(range);
    }

    return result;
}

static BasicBlockRange findBasicBlockAtTextOffset(int offset, const Vector<BasicBlockRange>& blocks)
{
    int bestDistance = INT_MAX;
    BasicBlockRange bestRange;
    bestRange.m_startOffset = bestRange.m_endOffset = -1;
    bestRange.m_hasExecuted = false; // Suppress MSVC warning.
    // Because some ranges may overlap because of function boundaries, make sure to find the smallest range enclosing the offset.
    for (BasicBlockRange range : blocks) {
        if (range.m_startOffset <= offset && offset <= range.m_endOffset && (range.m_endOffset - range.m_startOffset) < bestDistance) {
            RELEASE_ASSERT(range.m_endOffset - range.m_startOffset >= 0);
            bestDistance = range.m_endOffset - range.m_startOffset;
            bestRange = range;
        }
    }

    RELEASE_ASSERT(bestRange.m_startOffset != -1 && bestRange.m_endOffset != -1);
    return bestRange;
}

bool ControlFlowProfiler::hasBasicBlockAtTextOffsetBeenExecuted(int offset, SourceID sourceID, VM& vm)
{
    const Vector<BasicBlockRange>& blocks = getBasicBlocksForSourceID(sourceID, vm);
    BasicBlockRange range = findBasicBlockAtTextOffset(offset, blocks);
    return range.m_hasExecuted;
}

size_t ControlFlowProfiler::basicBlockExecutionCountAtTextOffset(int offset, SourceID sourceID, VM& vm)
{
    const Vector<BasicBlockRange>& blocks = getBasicBlocksForSourceID(sourceID, vm);
    BasicBlockRange range = findBasicBlockAtTextOffset(offset, blocks);
    return range.m_executionCount;
}

} // namespace JSC
