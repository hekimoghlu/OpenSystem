/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 5, 2023.
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
#include <wtf/TimingScope.h>

#include <wtf/HashMap.h>
#include <wtf/Lock.h>

namespace WTF {

namespace {

class State {
    WTF_MAKE_NONCOPYABLE(State);
    WTF_MAKE_FAST_ALLOCATED;
public:

    struct CallData {
        Seconds totalDuration;
        unsigned callCount { 0 };
        Seconds maxDuration;
        
        Seconds meanDuration() const { return totalDuration / callCount; }
    };

    State() = default;
    
    const CallData& addToTotal(const char* name, Seconds duration)
    {
        Locker locker { lock };
        auto& result = totals.add(name, CallData()).iterator->value;
        ++result.callCount;
        result.maxDuration = std::max(result.maxDuration, duration);
        result.totalDuration += duration;
        return result;
    }

private:
    HashMap<const char*, CallData> totals WTF_GUARDED_BY_LOCK(lock);
    Lock lock;
};

State& state()
{
    static Atomic<State*> s_state;
    return ensurePointer(s_state, [] { return new State(); });
}

} // anonymous namespace

void TimingScope::scopeDidEnd()
{
    const auto& data = state().addToTotal(m_name, MonotonicTime::now() - m_startTime);
    if (!(data.callCount % m_logIterationInterval))
        WTFLogAlways("%s: %u calls, mean duration: %.6fms, total duration: %.6fms, max duration %.6fms", m_name.characters(), data.callCount, data.meanDuration().milliseconds(), data.totalDuration.milliseconds(), data.maxDuration.milliseconds());
}

} // namespace WebCore
