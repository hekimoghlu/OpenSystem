/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#pragma once

#include "Options.h"
#include <wtf/Nonmovable.h>
#include <wtf/PrintStream.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {

class CodeBlock;

enum CountingVariant {
    CountingForBaseline,
    CountingForUpperTiers
};

double applyMemoryUsageHeuristics(int32_t value, CodeBlock*);
int32_t applyMemoryUsageHeuristicsAndConvertToInt(int32_t value, CodeBlock*);
int32_t maximumExecutionCountsBetweenCheckpoints(CountingVariant, CodeBlock*);

inline int32_t formattedTotalExecutionCount(float value)
{
    union {
        int32_t i;
        float f;
    } u;
    u.f = value;
    return u.i;
}

template<CountingVariant countingVariant>
class ExecutionCounter {
    WTF_MAKE_TZONE_ALLOCATED_TEMPLATE(ExecutionCounter);
    WTF_MAKE_NONMOVABLE(ExecutionCounter);
public:
    ExecutionCounter();
    void forceSlowPathConcurrently(); // If you use this, checkIfThresholdCrossedAndSet() may still return false.
    bool checkIfThresholdCrossedAndSet(CodeBlock*);
    void setNewThreshold(int32_t threshold, CodeBlock* = nullptr);
    void deferIndefinitely();
    double count() const { return static_cast<double>(m_totalCount) + m_counter; }
    void dump(PrintStream&) const;

    template<typename T>
    static T clippedThreshold(CodeBlock* codeBlock, T threshold)
    {
        int32_t maxThreshold = maximumExecutionCountsBetweenCheckpoints(countingVariant, codeBlock);
        if (threshold > maxThreshold)
            threshold = maxThreshold;
        return threshold;
    }

private:
    bool hasCrossedThreshold(CodeBlock*) const;
    bool setThreshold(CodeBlock*);
    void reset();

public:
    // NB. These are intentionally public because it will be modified from machine code.
    
    // This counter is incremented by the JIT or LLInt. It starts out negative and is
    // counted up until it becomes non-negative. At the start of a counting period,
    // the threshold we wish to reach is m_totalCount + m_counter, in the sense that
    // we will add X to m_totalCount and subtract X from m_counter.
    int32_t m_counter;

    // Counts the total number of executions we have seen plus the ones we've set a
    // threshold for in m_counter. Because m_counter's threshold is negative, the
    // total number of actual executions can always be computed as m_totalCount +
    // m_counter.
    float m_totalCount;

    // This is the threshold we were originally targeting, without any correction for
    // the memory usage heuristics.
    int32_t m_activeThreshold;
};

WTF_MAKE_TZONE_ALLOCATED_TEMPLATE_IMPL(template<CountingVariant countingVariant>, ExecutionCounter<countingVariant>);

typedef ExecutionCounter<CountingForBaseline> BaselineExecutionCounter;
typedef ExecutionCounter<CountingForUpperTiers> UpperTierExecutionCounter;

} // namespace JSC
