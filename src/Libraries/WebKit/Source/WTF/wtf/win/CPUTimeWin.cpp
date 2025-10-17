/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 23, 2025.
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
#include <wtf/CPUTime.h>

#include <windows.h>

namespace WTF {

static Seconds fileTimeToSeconds(const FILETIME& fileTime)
{
    // https://msdn.microsoft.com/ja-jp/library/windows/desktop/ms683223(v=vs.85).aspx
    // "All times are expressed using FILETIME data structures. Such a structure contains
    // two 32-bit values that combine to form a 64-bit count of 100-nanosecond time units."

    const constexpr double hundredsOfNanosecondsPerSecond = (1000.0 * 1000.0 * 1000.0) / 100.0;

    // https://msdn.microsoft.com/ja-jp/library/windows/desktop/ms724284(v=vs.85).aspx
    // "It is not recommended that you add and subtract values from the FILETIME structure to obtain relative times.
    // Instead, you should copy the low- and high-order parts of the file time to a ULARGE_INTEGER structure,
    // perform 64-bit arithmetic on the QuadPart member, and copy the LowPart and HighPart members into the FILETIME structure."
    ULARGE_INTEGER durationIn100NS;
    memcpy(&durationIn100NS, &fileTime, sizeof(durationIn100NS));
    return Seconds { durationIn100NS.QuadPart / hundredsOfNanosecondsPerSecond };
}

std::optional<CPUTime> CPUTime::get()
{
    // https://msdn.microsoft.com/ja-jp/library/windows/desktop/ms683223(v=vs.85).aspx
    FILETIME creationTime;
    FILETIME exitTime;
    FILETIME kernelTime;
    FILETIME userTime;
    if (!::GetProcessTimes(::GetCurrentProcess(), &creationTime, &exitTime, &kernelTime, &userTime))
        return std::nullopt;

    return CPUTime { MonotonicTime::now(), fileTimeToSeconds(userTime), fileTimeToSeconds(kernelTime) };
}

Seconds CPUTime::forCurrentThread()
{
    FILETIME userTime, kernelTime, creationTime, exitTime;

    BOOL ret = GetThreadTimes(GetCurrentThread(), &creationTime, &exitTime, &kernelTime, &userTime);
    RELEASE_ASSERT(ret);

    return fileTimeToSeconds(userTime) + fileTimeToSeconds(kernelTime);
}

}
