/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include "BasicBlockLocation.h"

#include "CCallHelpers.h"
#include <climits>
#include <wtf/DataLog.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BasicBlockLocation);

BasicBlockLocation::BasicBlockLocation(int startOffset, int endOffset)
    : m_startOffset(startOffset)
    , m_endOffset(endOffset)
    , m_executionCount(0)
{
}

void BasicBlockLocation::insertGap(int startOffset, int endOffset)
{
    std::pair<int, int> gap(startOffset, endOffset);
    if (!m_gaps.contains(gap))
        m_gaps.append(gap);
}

Vector<std::pair<int, int>> BasicBlockLocation::getExecutedRanges() const
{
    Vector<Gap> result;
    Vector<Gap> gaps = m_gaps;
    int nextRangeStart = m_startOffset;
    while (gaps.size()) {
        Gap minGap(INT_MAX, 0);
        unsigned minIdx = std::numeric_limits<unsigned>::max();
        for (unsigned idx = 0; idx < gaps.size(); idx++) {
            // Because we know that the Gaps inside m_gaps aren't enclosed within one another, it suffices to just check the first element to test ordering.
            if (gaps[idx].first < minGap.first) {
                minGap = gaps[idx];
                minIdx = idx;
            }
        }
        result.append(Gap(nextRangeStart, minGap.first - 1));
        nextRangeStart = minGap.second + 1;
        gaps.remove(minIdx);
    }

    result.append(Gap(nextRangeStart, m_endOffset));
    return result;
}

void BasicBlockLocation::dumpData() const
{
    Vector<Gap> executedRanges = getExecutedRanges();
    for (Gap gap : executedRanges) {
        dataLogF("\tBasicBlock: [%d, %d] hasExecuted: %s, executionCount:", gap.first, gap.second, hasExecuted() ? "true" : "false");
        dataLogLn(m_executionCount);
    }
}

#if ENABLE(JIT)
#if USE(JSVALUE64)
void BasicBlockLocation::emitExecuteCode(CCallHelpers& jit) const
{
    static_assert(sizeof(UCPURegister) == 8, "Assuming size_t is 64 bits on 64 bit platforms.");
    jit.add64(CCallHelpers::TrustedImm32(1), CCallHelpers::AbsoluteAddress(&m_executionCount));
}
#else
void BasicBlockLocation::emitExecuteCode(CCallHelpers& jit, MacroAssembler::RegisterID scratch) const
{
    static_assert(sizeof(size_t) == 4, "Assuming size_t is 32 bits on 32 bit platforms.");
    jit.load32(&m_executionCount, scratch);
    CCallHelpers::Jump done = jit.branchAdd32(CCallHelpers::Zero, scratch, CCallHelpers::TrustedImm32(1), scratch);
    jit.store32(scratch, std::bit_cast<void*>(&m_executionCount));
    done.link(&jit);
}
#endif // USE(JSVALUE64)
#endif // ENABLE(JIT)

} // namespace JSC
