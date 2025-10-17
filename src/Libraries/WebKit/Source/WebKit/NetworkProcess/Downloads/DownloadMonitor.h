/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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

#include <WebCore/Timer.h>
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class Download;

class DownloadMonitor {
    WTF_MAKE_TZONE_ALLOCATED(DownloadMonitor);
    WTF_MAKE_NONCOPYABLE(DownloadMonitor);
public:
    DownloadMonitor(Download&);
    
    void applicationDidEnterBackground();
    void applicationWillEnterForeground();
    void downloadReceivedBytes(uint64_t);
    void timerFired();

    void ref() const;
    void deref() const;

private:
    WeakRef<Download> m_download;

    double measuredThroughputRate() const;
    uint32_t testSpeedMultiplier() const;
    
    struct Timestamp {
        MonotonicTime time;
        uint64_t bytesReceived;
    };
    static constexpr size_t timestampCapacity = 10;
    Deque<Timestamp, timestampCapacity> m_timestamps;
    WebCore::Timer m_timer { *this, &DownloadMonitor::timerFired };
    size_t m_interval { 0 };
};

} // namespace WebKit
