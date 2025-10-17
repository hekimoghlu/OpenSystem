/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 15, 2022.
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
#include <wtf/RefCountedLeakCounter.h>

#include <wtf/Logging.h>

#ifdef NDEBUG
namespace WTF {

void RefCountedLeakCounter::suppressMessages(const char*) { }
void RefCountedLeakCounter::cancelMessageSuppression(const char*) { }

RefCountedLeakCounter::RefCountedLeakCounter(const char*) { }
RefCountedLeakCounter::~RefCountedLeakCounter() = default;

void RefCountedLeakCounter::increment() { }
void RefCountedLeakCounter::decrement() { }

} // namespace WTF
#else
#include <wtf/HashCountedSet.h>

namespace WTF {

typedef HashCountedSet<const char*, PtrHash<const char*>> ReasonSet;
static ReasonSet* leakMessageSuppressionReasons;

void RefCountedLeakCounter::suppressMessages(const char* reason)
{
    if (!leakMessageSuppressionReasons)
        leakMessageSuppressionReasons = new ReasonSet;
    leakMessageSuppressionReasons->add(reason);
}

void RefCountedLeakCounter::cancelMessageSuppression(const char* reason)
{
    ASSERT(leakMessageSuppressionReasons);
    ASSERT(leakMessageSuppressionReasons->contains(reason));
    leakMessageSuppressionReasons->remove(reason);
}

RefCountedLeakCounter::RefCountedLeakCounter(const char* description)
#if !LOG_DISABLED
    : m_description(description)
#endif
{
#if LOG_DISABLED
    UNUSED_PARAM(description);
#endif
}

RefCountedLeakCounter::~RefCountedLeakCounter()
{
    static bool loggedSuppressionReason;
    if (m_count) {
        if (!leakMessageSuppressionReasons || leakMessageSuppressionReasons->isEmpty())
            LOG(RefCountedLeaks, "LEAK: %u %s", m_count.load(), m_description);
        else if (!loggedSuppressionReason) {
            // This logs only one reason. Later we could change it so we log all the reasons.
            LOG(RefCountedLeaks, "No leak checking done: %s", leakMessageSuppressionReasons->begin()->key);
            loggedSuppressionReason = true;
        }
    }
}

void RefCountedLeakCounter::increment()
{
    ++m_count;
}

void RefCountedLeakCounter::decrement()
{
    --m_count;
}

} // namespace WTF
#endif
