/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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
#include <wtf/NumberOfCores.h>

#include <cstdio>
#include <mutex>

#if OS(DARWIN)
#include <sys/sysctl.h>
#elif OS(LINUX) || OS(AIX) || OS(OPENBSD) || OS(NETBSD) || OS(FREEBSD) || OS(HAIKU)
#include <unistd.h>
#elif OS(WINDOWS)
#include <windows.h>
#endif

namespace WTF {

int numberOfProcessorCores()
{
    const int defaultIfUnavailable = 1;
    static int s_numberOfCores = -1;

    if (s_numberOfCores > 0)
        return s_numberOfCores;
    
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    if (const char* coresEnv = getenv("WTF_numberOfProcessorCores")) {
        unsigned numberOfCores;
        if (sscanf(coresEnv, "%u", &numberOfCores) == 1) {
            s_numberOfCores = numberOfCores;
            return s_numberOfCores;
        } else
            fprintf(stderr, "WARNING: failed to parse WTF_numberOfProcessorCores=%s\n", coresEnv);
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

#if OS(DARWIN)
    unsigned result;
    size_t length = sizeof(result);
    int name[] = {
            CTL_HW,
            HW_AVAILCPU
    };
    int sysctlResult = sysctl(name, sizeof(name) / sizeof(int), &result, &length, 0, 0);

    s_numberOfCores = sysctlResult < 0 ? defaultIfUnavailable : result;
#elif OS(LINUX) || OS(AIX) || OS(OPENBSD) || OS(NETBSD) || OS(FREEBSD) || OS(HAIKU)
    long sysconfResult = sysconf(_SC_NPROCESSORS_ONLN);

    s_numberOfCores = sysconfResult < 0 ? defaultIfUnavailable : static_cast<int>(sysconfResult);
#elif OS(WINDOWS)
    UNUSED_PARAM(defaultIfUnavailable);
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);

    s_numberOfCores = sysInfo.dwNumberOfProcessors;
#else
    s_numberOfCores = defaultIfUnavailable;
#endif
    return s_numberOfCores;
}

#if OS(DARWIN)
int numberOfPhysicalProcessorCores()
{
    const int32_t defaultIfUnavailable = 1;

    static int32_t numCores = 0;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        size_t valueSize = sizeof(numCores);
        int result = sysctlbyname("hw.physicalcpu_max", &numCores, &valueSize, nullptr, 0);
        if (result < 0)
            numCores = defaultIfUnavailable;
    });

    return numCores;
}
#endif

}
