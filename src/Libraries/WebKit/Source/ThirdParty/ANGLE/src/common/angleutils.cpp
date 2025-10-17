/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 28, 2023.
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

//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "common/angleutils.h"
#include "common/debug.h"

#include <stdio.h>

#include <limits>
#include <vector>

namespace angle
{
// dirtyPointer is a special value that will make the comparison with any valid pointer fail and
// force the renderer to re-apply the state.
const uintptr_t DirtyPointer = std::numeric_limits<uintptr_t>::max();

// AMD_performance_monitor helpers.

PerfMonitorCounter::PerfMonitorCounter() = default;

PerfMonitorCounter::~PerfMonitorCounter() = default;

PerfMonitorCounterGroup::PerfMonitorCounterGroup() = default;

PerfMonitorCounterGroup::~PerfMonitorCounterGroup() = default;

uint32_t GetPerfMonitorCounterIndex(const PerfMonitorCounters &counters, const std::string &name)
{
    for (uint32_t counterIndex = 0; counterIndex < static_cast<uint32_t>(counters.size());
         ++counterIndex)
    {
        if (counters[counterIndex].name == name)
        {
            return counterIndex;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

uint32_t GetPerfMonitorCounterGroupIndex(const PerfMonitorCounterGroups &groups,
                                         const std::string &name)
{
    for (uint32_t groupIndex = 0; groupIndex < static_cast<uint32_t>(groups.size()); ++groupIndex)
    {
        if (groups[groupIndex].name == name)
        {
            return groupIndex;
        }
    }

    return std::numeric_limits<uint32_t>::max();
}

const PerfMonitorCounter &GetPerfMonitorCounter(const PerfMonitorCounters &counters,
                                                const std::string &name)
{
    return GetPerfMonitorCounter(const_cast<PerfMonitorCounters &>(counters), name);
}

PerfMonitorCounter &GetPerfMonitorCounter(PerfMonitorCounters &counters, const std::string &name)
{
    uint32_t counterIndex = GetPerfMonitorCounterIndex(counters, name);
    ASSERT(counterIndex < static_cast<uint32_t>(counters.size()));
    return counters[counterIndex];
}

const PerfMonitorCounterGroup &GetPerfMonitorCounterGroup(const PerfMonitorCounterGroups &groups,
                                                          const std::string &name)
{
    return GetPerfMonitorCounterGroup(const_cast<PerfMonitorCounterGroups &>(groups), name);
}

PerfMonitorCounterGroup &GetPerfMonitorCounterGroup(PerfMonitorCounterGroups &groups,
                                                    const std::string &name)
{
    uint32_t groupIndex = GetPerfMonitorCounterGroupIndex(groups, name);
    ASSERT(groupIndex < static_cast<uint32_t>(groups.size()));
    return groups[groupIndex];
}
}  // namespace angle

std::string ArrayString(unsigned int i)
{
    // We assume that UINT_MAX and GL_INVALID_INDEX are equal.
    ASSERT(i != UINT_MAX);

    std::stringstream strstr;
    strstr << "[";
    strstr << i;
    strstr << "]";
    return strstr.str();
}

std::string ArrayIndexString(const std::vector<unsigned int> &indices)
{
    std::stringstream strstr;

    for (auto indicesIt = indices.rbegin(); indicesIt != indices.rend(); ++indicesIt)
    {
        // We assume that UINT_MAX and GL_INVALID_INDEX are equal.
        ASSERT(*indicesIt != UINT_MAX);
        strstr << "[";
        strstr << (*indicesIt);
        strstr << "]";
    }

    return strstr.str();
}

size_t FormatStringIntoVector(const char *fmt, va_list vararg, std::vector<char> &outBuffer)
{
    va_list varargCopy;
    va_copy(varargCopy, vararg);

    int len = vsnprintf(nullptr, 0, fmt, vararg);
    ASSERT(len >= 0);

    outBuffer.resize(len + 1, 0);

    len = vsnprintf(outBuffer.data(), outBuffer.size(), fmt, varargCopy);
    va_end(varargCopy);
    ASSERT(len >= 0);
    return static_cast<size_t>(len);
}
