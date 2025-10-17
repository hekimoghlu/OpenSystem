/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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

#include <wtf/ApproximateTime.h>
#include <wtf/HashMap.h>
#include <wtf/PriorityQueue.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class ResourceMonitorThrottler final {
    WTF_MAKE_FAST_ALLOCATED;
public:
    WEBCORE_EXPORT ResourceMonitorThrottler();
    WEBCORE_EXPORT ResourceMonitorThrottler(size_t count, Seconds duration, size_t maxHosts);

    WEBCORE_EXPORT bool tryAccess(const String& host, ApproximateTime = ApproximateTime::now());

    WEBCORE_EXPORT void setCountPerDuration(size_t, Seconds);

private:
    struct Config {
        size_t count;
        Seconds duration;
        size_t maxHosts;
    };

    class AccessThrottler final {
    public:
        AccessThrottler() = default;

        bool tryAccessAndUpdateHistory(ApproximateTime, const Config&);
        bool tryExpire(ApproximateTime, const Config&);
        ApproximateTime oldestAccessTime() const;
        ApproximateTime newestAccessTime() const { return m_newestAccessTime; }

    private:
        void removeExpired(ApproximateTime);

        PriorityQueue<ApproximateTime> m_accessTimes;
        ApproximateTime m_newestAccessTime { -ApproximateTime::infinity() };
    };

    AccessThrottler& throttlerForHost(const String& host);
    void removeExpiredThrottler();
    void removeOldestThrottler();

    Config m_config;
    HashMap<String, AccessThrottler> m_throttlersByHost;
};

}
