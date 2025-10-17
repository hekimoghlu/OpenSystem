/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

#if OS(LINUX)

#include <wtf/NeverDestroyed.h>
#include <wtf/Noncopyable.h>
#include <wtf/text/CString.h>

namespace WebKit {

class MemoryPressureMonitor {
    WTF_MAKE_NONCOPYABLE(MemoryPressureMonitor);
    friend NeverDestroyed<MemoryPressureMonitor>;
public:
    static MemoryPressureMonitor& singleton();
    void start();
    static bool disabled();

    ~MemoryPressureMonitor();

private:
    MemoryPressureMonitor() = default;
    bool m_started { false };
    static bool s_disabled;
};

class CGroupMemoryController {
public:
    CGroupMemoryController() = default;
    bool isActive() { return !m_cgroupMemoryControllerPath.isNull(); };

    void setMemoryControllerPath(CString);

    size_t getMemoryTotalWithCgroup();
    size_t getMemoryUsageWithCgroup();

    ~CGroupMemoryController()
    {
        disposeMemoryController();
    }

private:
    CString m_cgroupMemoryControllerPath;

    FILE* m_cgroupMemoryMemswLimitInBytesFile { nullptr };
    FILE* m_cgroupMemoryMemswUsageInBytesFile { nullptr };
    FILE* m_cgroupMemoryLimitInBytesFile { nullptr };
    FILE* m_cgroupMemoryUsageInBytesFile { nullptr };

    FILE* m_cgroupV2MemoryMemswMaxFile { nullptr };
    FILE* m_cgroupV2MemoryMaxFile { nullptr };
    FILE* m_cgroupV2MemoryHighFile { nullptr };
    FILE* m_cgroupV2MemoryCurrentFile { nullptr };

    void disposeMemoryController();
    size_t getCgroupFileValue(FILE*);
};

} // namespace WebKit

#endif // OS(LINUX)
