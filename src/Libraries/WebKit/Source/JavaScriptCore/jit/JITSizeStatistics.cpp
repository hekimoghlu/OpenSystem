/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#include "JITSizeStatistics.h"

#if ENABLE(JIT)

#include "CCallHelpers.h"
#include "JITPlan.h"
#include "LinkBuffer.h"
#include <wtf/BubbleSort.h>
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(JITSizeStatistics);

JITSizeStatistics::Marker JITSizeStatistics::markStart(String identifier, CCallHelpers& jit)
{
    Marker marker;
    marker.identifier = identifier;
    marker.start = jit.labelIgnoringWatchpoints();
    return marker;
}

void JITSizeStatistics::markEnd(Marker marker, CCallHelpers& jit, JITPlan& planRef)
{
    CCallHelpers::Label end = jit.labelIgnoringWatchpoints();
    auto* plan = &planRef;
    jit.addLinkTask([=, this] (LinkBuffer& linkBuffer) {
        size_t size = linkBuffer.locationOf<NoPtrTag>(end).untaggedPtr<char*>() - linkBuffer.locationOf<NoPtrTag>(marker.start).untaggedPtr<char*>();
        plan->addMainThreadFinalizationTask([=, this] {
            auto& entry = m_data.add(marker.identifier, Entry { }).iterator->value;
            ++entry.count;
            entry.totalBytes += size;
        });
    });
}

void JITSizeStatistics::dump(PrintStream& out) const
{
    Vector<std::pair<String, Entry>> entries;

    for (auto pair : m_data)
        entries.append(std::make_pair(pair.key, pair.value));

    std::sort(entries.begin(), entries.end(), [] (const auto& lhs, const auto& rhs) {
        return lhs.second.totalBytes > rhs.second.totalBytes;
    });

    out.println("JIT size statistics:");
    out.println("==============================================");

    for (auto& entry : entries) {
        size_t totalBytes = entry.second.totalBytes;
        size_t count = entry.second.count;
        double avg = static_cast<double>(totalBytes) / static_cast<double>(count);
        out.println(entry.first, " totalBytes: ", totalBytes, " count: ", count, " avg: ", avg);
    }
}

} // namespace JSC

#endif // ENABLE(JIT)
