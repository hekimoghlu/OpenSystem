/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 18, 2022.
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
#include "DownloadMonitor.h"

#include "Download.h"
#include "Logging.h"
#include <wtf/TZoneMallocInlines.h>

#define DOWNLOAD_MONITOR_RELEASE_LOG(fmt, ...) RELEASE_LOG(Network, "%p - DownloadMonitor::" fmt, this, ##__VA_ARGS__)

namespace WebKit {

constexpr uint64_t operator""_kbps(unsigned long long kilobytesPerSecond)
{
    return kilobytesPerSecond * 1024;
}

struct ThroughputInterval {
    Seconds time;
    uint64_t bytesPerSecond;
};

static constexpr std::array throughputIntervals = {
    ThroughputInterval { 1_min, 1_kbps },
    ThroughputInterval { 5_min, 2_kbps },
    ThroughputInterval { 10_min, 4_kbps },
    ThroughputInterval { 15_min, 8_kbps },
    ThroughputInterval { 20_min, 16_kbps },
    ThroughputInterval { 25_min, 32_kbps },
    ThroughputInterval { 30_min, 64_kbps },
    ThroughputInterval { 45_min, 96_kbps },
    ThroughputInterval { 60_min, 128_kbps }
};

static Seconds timeUntilNextInterval(size_t currentInterval)
{
    RELEASE_ASSERT(currentInterval + 1 < throughputIntervals.size());
    return throughputIntervals[currentInterval + 1].time - throughputIntervals[currentInterval].time;
}

DownloadMonitor::DownloadMonitor(Download& download)
    : m_download(download)
{
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(DownloadMonitor);

void DownloadMonitor::ref() const
{
    m_download->ref();
}

void DownloadMonitor::deref() const
{
    m_download->deref();
}

double DownloadMonitor::measuredThroughputRate() const
{
    uint64_t bytes { 0 };
    for (const auto& timestamp : m_timestamps)
        bytes += timestamp.bytesReceived;
    if (!bytes)
        return 0;
    ASSERT(!m_timestamps.isEmpty());
    Seconds timeDifference = m_timestamps.last().time.secondsSinceEpoch() - m_timestamps.first().time.secondsSinceEpoch();
    double seconds = timeDifference.seconds();
    if (!seconds)
        return 0;
    return bytes / seconds;
}

void DownloadMonitor::downloadReceivedBytes(uint64_t bytesReceived)
{
    if (m_timestamps.size() > timestampCapacity - 1) {
        ASSERT(m_timestamps.size() == timestampCapacity);
        m_timestamps.removeFirst();
    }
    m_timestamps.append({ MonotonicTime::now(), bytesReceived });
}

void DownloadMonitor::applicationWillEnterForeground()
{
    DOWNLOAD_MONITOR_RELEASE_LOG("applicationWillEnterForeground (id = %" PRIu64 ")", m_download->downloadID().toUInt64());
    m_timer.stop();
    m_interval = 0;
}

void DownloadMonitor::applicationDidEnterBackground()
{
    DOWNLOAD_MONITOR_RELEASE_LOG("applicationDidEnterBackground (id = %" PRIu64 ")", m_download->downloadID().toUInt64());
    ASSERT(!m_timer.isActive());
    ASSERT(!m_interval);
    m_timer.startOneShot(throughputIntervals[0].time / testSpeedMultiplier());
}

uint32_t DownloadMonitor::testSpeedMultiplier() const
{
    return m_download->testSpeedMultiplier();
}

void DownloadMonitor::timerFired()
{
    downloadReceivedBytes(0);

    RELEASE_ASSERT(m_interval < std::size(throughputIntervals));
    if (measuredThroughputRate() < throughputIntervals[m_interval].bytesPerSecond) {
        DOWNLOAD_MONITOR_RELEASE_LOG("timerFired: cancelling download (id = %" PRIu64 ")", m_download->downloadID().toUInt64());
        Ref { m_download.get() }->cancel([](auto) { }, Download::IgnoreDidFailCallback::No);
    } else if (m_interval + 1 < std::size(throughputIntervals)) {
        DOWNLOAD_MONITOR_RELEASE_LOG("timerFired: sufficient throughput rate (id = %" PRIu64 ")", m_download->downloadID().toUInt64());
        m_timer.startOneShot(timeUntilNextInterval(m_interval++) / testSpeedMultiplier());
    } else
        DOWNLOAD_MONITOR_RELEASE_LOG("timerFired: Download reached threshold to not be terminated (id = %" PRIu64 ")", m_download->downloadID().toUInt64());
}

} // namespace WebKit
