/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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
#include "DNSResolveQueue.h"

#if USE(GLIB)
#include "DNSResolveQueueGLib.h"
#elif USE(CURL)
#include "DNSResolveQueueCurl.h"
#elif USE(CF)
#include "DNSResolveQueueCFNet.h"
#endif

#include <wtf/CompletionHandler.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

// When resolve queue is empty, we fire async resolution requests immediately (which is important if the prefetch is triggered by hovering).
// But during page parsing, we should coalesce identical requests to avoid stressing out the DNS resolver.
static const int gNamesToResolveImmediately = 4;

// Coalesce prefetch requests for this long before sending them out.
static const Seconds coalesceDelay { 1_s };

// Sending many DNS requests at once can overwhelm some gateways. See <rdar://8105550> for specific CFNET issues with CFHost throttling.
static const int gMaxSimultaneousRequests = 8;

// For a page has links to many outside sites, it is likely that the system DNS resolver won't be able to cache them all anyway, and we don't want
// to negatively affect other applications' performance by pushing their cached entries out.
// If we end up with lots of names to prefetch, some will be dropped.
static const int gMaxRequestsToQueue = 64;

// If there were queued names that couldn't be sent simultaneously, check the state of resolvers after this delay.
static const Seconds resolvingRetryDelay { 100_ms };

DNSResolveQueue& DNSResolveQueue::singleton()
{
    static NeverDestroyed<DNSResolveQueuePlatform> queue;

    return queue;
}

DNSResolveQueue::DNSResolveQueue()
    : m_timer(*this, &DNSResolveQueue::timerFired)
    , m_requestsInFlight(0)
{
    // isUsingProxy will return the initial value of m_isUsingProxy at first on
    // platforms that have an asynchronous implementation of updateIsUsingProxy,
    // so initialize it to true so we won't prefetch before we know if we are using a proxy.
}

// Don't do DNS prefetch if proxies are involved. For many proxy types, the user agent is never
// exposed to the IP address during normal operation. Querying an internal DNS server may not help
// performance, as it doesn't necessarily look up the actual external IP. Also, if DNS returns a
// fake internal address, local caches may keep it even after re-connecting to another network.
bool DNSResolveQueue::isUsingProxy()
{
    MonotonicTime time = MonotonicTime::now();
    static const Seconds minimumProxyCheckDelay = 5_s;
    if (time - m_lastProxyEnabledStatusCheckTime > minimumProxyCheckDelay) {
        m_lastProxyEnabledStatusCheckTime = time;
        updateIsUsingProxy();
    }
    return m_isUsingProxy;
}

void DNSResolveQueue::add(const String& hostname)
{
    // If there are no names queued, and few enough are in flight, resolve immediately (the mouse may be over a link).
    if (!m_names.size()) {
        if (isUsingProxy())
            return;
        if (++m_requestsInFlight <= gNamesToResolveImmediately) {
            platformResolve(hostname);
            return;
        }
        --m_requestsInFlight;
    }

    // It's better to not prefetch some names than to clog the queue.
    // Dropping the newest names, because on a single page, these are likely to be below oldest ones.
    if (m_names.size() < gMaxRequestsToQueue) {
        m_names.add(hostname);
        if (!m_timer.isActive())
            m_timer.startOneShot(coalesceDelay);
    }
}

void DNSResolveQueue::timerFired()
{
    if (isUsingProxy()) {
        m_names.clear();
        return;
    }

    int requestsAllowed = gMaxSimultaneousRequests - m_requestsInFlight;

    for (; !m_names.isEmpty() && requestsAllowed > 0; --requestsAllowed) {
        ++m_requestsInFlight;
        UncheckedKeyHashSet<String>::iterator currentName = m_names.begin();
        platformResolve(*currentName);
        m_names.remove(currentName);
    }

    if (!m_names.isEmpty())
        m_timer.startOneShot(resolvingRetryDelay);
}

} // namespace WebCore
