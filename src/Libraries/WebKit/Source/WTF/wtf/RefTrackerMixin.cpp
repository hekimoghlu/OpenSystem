/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 4, 2023.
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
#include "RefTrackerMixin.h"

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

#if ENABLE(REFTRACKER)

void RefTracker::reportLive(void* id)
{
    RELEASE_ASSERT(id);
    Locker locker(lock);

    std::unique_ptr<StackShot> stack = nullptr;
    if (!loggingDisabledDepth.load())
        stack = makeUnique<StackShot>(16);
    RELEASE_ASSERT(map.add(id, WTFMove(stack)).isNewEntry);
}

void RefTracker::reportDead(void* id)
{
    RELEASE_ASSERT(id);
    Locker locker(lock);
    if (!map.contains(id)) {
        WTFReportBacktrace();
        WTFLogAlways("******************************************** Dead RefTracker was never live: %p", id);
    }
    RELEASE_ASSERT(map.contains(id));
    RELEASE_ASSERT(map.remove(id));
}

void RefTracker::logAllLiveReferences()
{
    static constexpr int framesToSkip = 3;
    Locker locker(lock);
    auto keysIterator = map.keys();
    auto keys = std::vector<void*> { keysIterator.begin(), keysIterator.end() };
    std::sort(keys.begin(), keys.end());
    for (auto& k : keys) {
        auto* v = map.get(k);
        if (!v)
            continue;
        dataLogLn(StackTracePrinter { { v->array() + framesToSkip, v->size() - framesToSkip } });
        dataLogLn("\n---\n");
    }
}

#endif // ENABLE(REFTRACKER)

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
