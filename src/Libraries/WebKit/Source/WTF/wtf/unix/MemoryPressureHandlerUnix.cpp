/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#include <wtf/MemoryPressureHandler.h>

#include <malloc.h>
#include <unistd.h>
#include <wtf/Logging.h>
#include <wtf/MainThread.h>
#include <wtf/MemoryFootprint.h>
#include <wtf/text/WTFString.h>

#if OS(LINUX)
#include <wtf/linux/CurrentProcessMemoryStatus.h>
#elif OS(FREEBSD)
#include <sys/sysctl.h>
#include <sys/types.h>
#include <sys/user.h>
#elif OS(QNX)
#include <fcntl.h>
#include <sys/procfs.h>
#endif

namespace WTF {

// Disable memory event reception for a minimum of s_minimumHoldOffTime
// seconds after receiving an event. Don't let events fire any sooner than
// s_holdOffMultiplier times the last cleanup processing time. Effectively
// this is 1 / s_holdOffMultiplier percent of the time.
// If after releasing the memory we don't free at least s_minimumBytesFreedToUseMinimumHoldOffTime,
// we wait longer to try again (s_maximumHoldOffTime).
// These value seems reasonable and testing verifies that it throttles frequent
// low memory events, greatly reducing CPU usage.
static const Seconds s_minimumHoldOffTime { 5_s };
static const Seconds s_maximumHoldOffTime { 30_s };
static const size_t s_minimumBytesFreedToUseMinimumHoldOffTime = 1 * MB;
static const unsigned s_holdOffMultiplier = 20;

void MemoryPressureHandler::triggerMemoryPressureEvent(bool isCritical)
{
    if (!m_installed)
        return;

    if (ReliefLogger::loggingEnabled())
        LOG(MemoryPressure, "Got memory pressure notification (%s)", isCritical ? "critical" : "non-critical");

    setMemoryPressureStatus(SystemMemoryPressureStatus::Critical);

    ensureOnMainThread([this, isCritical] {
        respondToMemoryPressure(isCritical ? Critical::Yes : Critical::No);
    });

    if (ReliefLogger::loggingEnabled() && isUnderMemoryPressure())
        LOG(MemoryPressure, "System is no longer under memory pressure.");

    setMemoryPressureStatus(SystemMemoryPressureStatus::Normal);
}

void MemoryPressureHandler::install()
{
    if (m_installed || m_holdOffTimer.isActive())
        return;

    m_installed = true;
}

void MemoryPressureHandler::uninstall()
{
    if (!m_installed)
        return;

    m_holdOffTimer.stop();

    m_installed = false;
}

void MemoryPressureHandler::holdOffTimerFired()
{
    install();
}

void MemoryPressureHandler::holdOff(Seconds seconds)
{
    m_holdOffTimer.startOneShot(seconds);
}

static size_t processMemoryUsage()
{
#if OS(LINUX)
    ProcessMemoryStatus memoryStatus;
    currentProcessMemoryStatus(memoryStatus);
    return (memoryStatus.resident - memoryStatus.shared);
#elif OS(FREEBSD)
    static size_t pageSize = sysconf(_SC_PAGE_SIZE);
    struct kinfo_proc info;
    size_t infolen = sizeof(info);

    int mib[4];
    mib[0] = CTL_KERN;
    mib[1] = KERN_PROC;
    mib[2] = KERN_PROC_PID;
    mib[3] = getpid();

    if (sysctl(mib, 4, &info, &infolen, nullptr, 0))
        return 0;

    return static_cast<size_t>(info.ki_rssize - info.ki_tsize) * pageSize;
#elif OS(QNX)
    int fd = open("/proc/self/ctl", O_RDONLY);
    if (fd == -1)
        return 0;

    procfs_asinfo info;
    int rc = devctl(fd, DCMD_PROC_ASINFO, &info, sizeof(info), nullptr);
    close(fd);

    if (rc)
        return 0;

    return static_cast<size_t>(info.rss);
#else
#error "Missing a platform specific way of determining the memory usage"
#endif
}

void MemoryPressureHandler::respondToMemoryPressure(Critical critical, Synchronous synchronous)
{
    uninstall();

    MonotonicTime startTime = MonotonicTime::now();
    int64_t processMemory = processMemoryUsage();
    releaseMemory(critical, synchronous);
    int64_t bytesFreed = processMemory - processMemoryUsage();
    Seconds holdOffTime = s_maximumHoldOffTime;
    if (bytesFreed > 0 && static_cast<size_t>(bytesFreed) >= s_minimumBytesFreedToUseMinimumHoldOffTime)
        holdOffTime = (MonotonicTime::now() - startTime) * s_holdOffMultiplier;
    holdOff(std::max(holdOffTime, s_minimumHoldOffTime));
}

void MemoryPressureHandler::platformReleaseMemory(Critical)
{
#if HAVE(MALLOC_TRIM)
    malloc_trim(0);
#endif
}

std::optional<MemoryPressureHandler::ReliefLogger::MemoryUsage> MemoryPressureHandler::ReliefLogger::platformMemoryUsage()
{
    return MemoryUsage {processMemoryUsage(), memoryFootprint()};
}

} // namespace WTF
