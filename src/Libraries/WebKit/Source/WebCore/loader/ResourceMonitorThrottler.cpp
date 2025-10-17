/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "ResourceMonitorThrottler.h"

#include "Logging.h"
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/Seconds.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/StringHash.h>

#if ENABLE(CONTENT_EXTENSIONS)

#define RESOURCEMONITOR_RELEASE_LOG(fmt, ...) RELEASE_LOG(ResourceLoading, "%p - ResourceMonitorThrottler::" fmt, this, ##__VA_ARGS__)

namespace WebCore {

static constexpr size_t defaultThrottleAccessCount = 5;
static constexpr Seconds defaultThrottleDuration = 24_h;
static constexpr size_t defaultMaxHosts = 100;

ResourceMonitorThrottler::ResourceMonitorThrottler()
    : ResourceMonitorThrottler(defaultThrottleAccessCount, defaultThrottleDuration, defaultMaxHosts)
{
}

ResourceMonitorThrottler::ResourceMonitorThrottler(size_t count, Seconds duration, size_t maxHosts)
    : m_config { count, duration, maxHosts }
{
    ASSERT(maxHosts >= 1);
    RESOURCEMONITOR_RELEASE_LOG("Initialized with count: %zu, duration: %.f, maxHosts: %zu", count, duration.value(), maxHosts);
}

auto ResourceMonitorThrottler::throttlerForHost(const String& host) -> AccessThrottler&
{
    return m_throttlersByHost.ensure(host, [] {
        return AccessThrottler { };
    }).iterator->value;
}

void ResourceMonitorThrottler::removeOldestThrottler()
{
    auto oldest = ApproximateTime::infinity();
    String oldestKey;

    for (auto it : m_throttlersByHost) {
        auto time = it.value.newestAccessTime();
        if (time < oldest) {
            oldest = time;
            oldestKey = it.key;
        }
    }
    ASSERT(!oldestKey.isNull());
    m_throttlersByHost.remove(oldestKey);
}

bool ResourceMonitorThrottler::tryAccess(const String& host, ApproximateTime time)
{
    if (host.isEmpty())
        return false;

    auto& throttler = throttlerForHost(host);
    auto result = throttler.tryAccessAndUpdateHistory(time, m_config);

    if (m_throttlersByHost.size() > m_config.maxHosts) {
        // Update and remove all expired access times. If no entry in throttler, remove it.
        m_throttlersByHost.removeIf([&](auto& it) -> bool {
            return it.value.tryExpire(time, m_config);
        });

        // If there are still too many hosts, then remove oldest one.
        while (m_throttlersByHost.size() > m_config.maxHosts)
            removeOldestThrottler();
    }
    ASSERT(m_throttlersByHost.size() <= m_config.maxHosts);

    return result;
}

bool ResourceMonitorThrottler::AccessThrottler::tryAccessAndUpdateHistory(ApproximateTime time, const Config& config)
{
    tryExpire(time, config);
    if (m_accessTimes.size() >= config.count)
        return false;

    m_accessTimes.enqueue(time);
    if (m_newestAccessTime < time)
        m_newestAccessTime = time;

    return true;
}

ApproximateTime ResourceMonitorThrottler::AccessThrottler::oldestAccessTime() const
{
    return m_accessTimes.peek();
}

bool ResourceMonitorThrottler::AccessThrottler::tryExpire(ApproximateTime time, const Config& config)
{
    auto expirationTime = time - config.duration;

    while (!m_accessTimes.isEmpty()) {
        if (oldestAccessTime() > expirationTime)
            return false;

        m_accessTimes.dequeue();
    }
    // Tell caller that the queue is empty.
    return true;
}

void ResourceMonitorThrottler::setCountPerDuration(size_t count, Seconds duration)
{
    m_config.count = count;
    m_config.duration = duration;
}

} // namespace WebCore

#undef RESOURCEMONITOR_RELEASE_LOG

#endif
