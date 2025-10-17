/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#include "FunctionHasExecutedCache.h"

#include <limits.h>

namespace JSC {

bool FunctionHasExecutedCache::hasExecutedAtOffset(SourceID id, unsigned offset)
{
    auto iterator = m_rangeMap.find(id);
    if (iterator == m_rangeMap.end())
        return false;

    RangeMap& map = iterator->value;
    unsigned distance = UINT_MAX;
    bool hasExecuted = false;
    for (auto& pair : map) {
        const FunctionRange& range = pair.key.key();
        if (range.m_start <= offset && offset <= range.m_end && range.m_end - range.m_start < distance) {
            hasExecuted = pair.value;
            distance = range.m_end - range.m_start;
        }
    }

    return hasExecuted;
}

void FunctionHasExecutedCache::insertUnexecutedRange(SourceID id, unsigned start, unsigned end)
{
    RangeMap& map = m_rangeMap.add(id, RangeMap { }).iterator->value;
    FunctionRange range;
    range.m_start = start;
    range.m_end = end;
    // Only insert unexecuted ranges once for a given sourceID because we may run into a situation where an executable executes, then is GCed, and then is allocated again,
    // and tries to reinsert itself, claiming it has never run, but this is false because it indeed already executed.
    map.add(range, false);
}

void FunctionHasExecutedCache::removeUnexecutedRange(SourceID id, unsigned start, unsigned end)
{
    // FIXME: We should never have an instance where we return here, but currently do in some situations. Find out why.
    auto iterator = m_rangeMap.find(id);
    if (iterator == m_rangeMap.end())
        return;

    RangeMap& map = iterator->value;

    FunctionRange range;
    range.m_start = start;
    range.m_end = end;
    map.set(range, true);
}

Vector<std::tuple<bool, unsigned, unsigned>> FunctionHasExecutedCache::getFunctionRanges(SourceID id)
{
    Vector<std::tuple<bool, unsigned, unsigned>> ranges(0);
    auto iterator = m_rangeMap.find(id);
    if (iterator == m_rangeMap.end())
        return ranges;

    RangeMap& map = iterator->value;
    for (auto& pair : map) {
        const FunctionRange& range = pair.key.key();
        bool hasExecuted = pair.value;
        ranges.append(std::tuple<bool, unsigned, unsigned>(hasExecuted, range.m_start, range.m_end));
    }

    return ranges;
}

} // namespace JSC
