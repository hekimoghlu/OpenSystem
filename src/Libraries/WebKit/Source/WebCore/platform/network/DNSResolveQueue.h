/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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

#include "DNS.h"
#include "Timer.h"
#include <atomic>
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class DNSResolveQueue {
    friend NeverDestroyed<DNSResolveQueue>;

public:
    virtual ~DNSResolveQueue() = default;

    static DNSResolveQueue& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    virtual void resolve(const String& hostname, uint64_t identifier, DNSCompletionHandler&&) = 0;
    virtual void stopResolve(uint64_t identifier) = 0;
    void add(const String& hostname);
    void decrementRequestCount()
    {
        --m_requestsInFlight;
    }

protected:
    DNSResolveQueue();
    bool isUsingProxy();

    bool m_isUsingProxy { true };

private:
    virtual void updateIsUsingProxy() = 0;
    virtual void platformResolve(const String&) = 0;
    void timerFired();

    Timer m_timer;

    UncheckedKeyHashSet<String> m_names;
    std::atomic<int> m_requestsInFlight;
    MonotonicTime m_lastProxyEnabledStatusCheckTime;
};

}
