/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 13, 2023.
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
#include <wtf/RAMSize.h>

#include <mutex>

#if OS(WINDOWS)
#include <windows.h>
#elif USE(SYSTEM_MALLOC)
#if OS(LINUX) || OS(FREEBSD)
#include <sys/sysinfo.h>
#elif OS(UNIX) || OS(HAIKU)
#include <unistd.h>
#endif // OS(LINUX) || OS(FREEBSD) || OS(UNIX) || OS(HAIKU)
#else
#include <bmalloc/bmalloc.h>
#endif

#if OS(DARWIN)
#include <mach/mach.h>
#endif

namespace WTF {

#if OS(WINDOWS)
static constexpr size_t ramSizeGuess = 512 * MB;
#endif

static size_t computeRAMSize()
{
#if OS(WINDOWS)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    bool result = GlobalMemoryStatusEx(&status);
    if (!result)
        return ramSizeGuess;
    return status.ullTotalPhys;
#elif USE(SYSTEM_MALLOC)
#if OS(LINUX) || OS(FREEBSD)
    struct sysinfo si;
    sysinfo(&si);
    return si.totalram * si.mem_unit;
#elif OS(UNIX) || OS(HAIKU)
    long pages = sysconf(_SC_PHYS_PAGES);
    long pageSize = sysconf(_SC_PAGE_SIZE);
    return pages * pageSize;
#else
#error "Missing a platform specific way of determining the available RAM"
#endif // OS(LINUX) || OS(FREEBSD) || OS(UNIX) || OS(HAIKU)
#else
    return bmalloc::api::availableMemory();
#endif
}

size_t ramSize()
{
    static size_t ramSize;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        ramSize = computeRAMSize();
    });
    return ramSize;
}

#if OS(DARWIN)
size_t ramSizeDisregardingJetsamLimit()
{
    host_basic_info_data_t hostInfo;

    mach_port_t host = mach_host_self();
    mach_msg_type_number_t count = HOST_BASIC_INFO_COUNT;
    kern_return_t r = host_info(host, HOST_BASIC_INFO, (host_info_t)&hostInfo, &count);
    if (mach_port_deallocate(mach_task_self(), host) != KERN_SUCCESS)
        return 0;
    if (r != KERN_SUCCESS)
        return 0;

    if (hostInfo.max_mem > std::numeric_limits<size_t>::max())
        return std::numeric_limits<size_t>::max();

    return static_cast<size_t>(hostInfo.max_mem);
}
#endif

} // namespace WTF
