/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

#include <psapi.h>
#include <wtf/NeverDestroyed.h>

namespace WTF {

void MemoryPressureHandler::platformInitialize()
{
    m_lowMemoryHandle = Win32Handle::adopt(::CreateMemoryResourceNotification(LowMemoryResourceNotification));
}

void MemoryPressureHandler::windowsMeasurementTimerFired()
{
    setMemoryPressureStatus(SystemMemoryPressureStatus::Normal);

    BOOL memoryLow;

    if (QueryMemoryResourceNotification(m_lowMemoryHandle.get(), &memoryLow) && memoryLow) {
        setMemoryPressureStatus(SystemMemoryPressureStatus::Critical);
        releaseMemory(Critical::Yes);
        return;
    }

#if CPU(X86)
    PROCESS_MEMORY_COUNTERS_EX counters;

    if (!GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters)))
        return;

    // On Windows, 32-bit processes have 2GB of memory available, where some is used by the system.
    // Debugging has shown that allocations might fail and cause crashes when memory usage is > ~1GB.
    // We should start releasing memory before we reach 1GB.
    const int maxMemoryUsageBytes = 0.9 * 1024 * 1024 * 1024;

    if (counters.PrivateUsage > maxMemoryUsageBytes) {
        didExceedProcessMemoryLimit(ProcessMemoryLimit::Critical);
        releaseMemory(Critical::Yes);
    }
#endif
}

void MemoryPressureHandler::platformReleaseMemory(Critical)
{
}

void MemoryPressureHandler::install()
{
    m_installed = true;
    m_windowsMeasurementTimer.startRepeating(60_s);
}

void MemoryPressureHandler::uninstall()
{
    if (!m_installed)
        return;

    m_windowsMeasurementTimer.stop();
    m_installed = false;
}

void MemoryPressureHandler::holdOff(Seconds)
{
}

void MemoryPressureHandler::respondToMemoryPressure(Critical critical, Synchronous synchronous)
{
    uninstall();

    releaseMemory(critical, synchronous);
}

std::optional<MemoryPressureHandler::ReliefLogger::MemoryUsage> MemoryPressureHandler::ReliefLogger::platformMemoryUsage()
{
    return std::nullopt;
}

} // namespace WTF
