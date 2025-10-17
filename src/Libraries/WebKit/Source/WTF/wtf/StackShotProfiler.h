/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

#include <wtf/DataLog.h>
#include <wtf/Locker.h>
#include <wtf/ProcessID.h>
#include <wtf/Spectrum.h>
#include <wtf/StackTrace.h>
#include <wtf/StackShot.h>
#include <wtf/Threading.h>
#include <wtf/WordLock.h>

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN

namespace WTF {

class StackShotProfiler {
    WTF_MAKE_FAST_ALLOCATED;
public:
    StackShotProfiler(unsigned numFrames, unsigned framesToSkip, unsigned stacksToReport)
        : m_numFrames(numFrames)
        , m_framesToSkip(framesToSkip)
        , m_stacksToReport(stacksToReport)
    {
        Thread::create("StackShotProfiler"_s, [this] () { run(); });
    }

    // NEVER_INLINE so that framesToSkip is predictable.
    NEVER_INLINE void profile()
    {
        Locker locker { m_lock };
        m_profile.add(StackShot(m_numFrames + m_framesToSkip));
        m_totalCount++;
    }
    
private:
    NO_RETURN void run()
    {
        for (;;) {
            sleep(1_s);
            Locker locker { m_lock };
            {
                Locker spectrumLocker { m_profile.getLock() };
                auto list = m_profile.buildList(spectrumLocker);
                dataLog("\nHottest stacks in ", getCurrentProcessID(), ":\n");
                for (size_t i = list.size(), count = 0; i-- && count < m_stacksToReport; count++) {
                    auto& entry = list[i];
                    dataLog("\nTop #", count + 1, " stack: ", entry.count * 100 / m_totalCount, "%\n");
                    dataLog(StackTracePrinter { { entry.key->array() + m_framesToSkip, entry.key->size() - m_framesToSkip } });
                }
                dataLog("\n");
            }
        }
    }
    
    WordLock m_lock;
    Spectrum<StackShot, double> m_profile;
    double m_totalCount { 0 };
    unsigned m_numFrames;
    unsigned m_framesToSkip;
    unsigned m_stacksToReport;
};

#define STACK_SHOT_PROFILE(numFrames, framesToSkip, stacksToReport) do { \
    static StackShotProfiler* stackShotProfiler; \
    static std::once_flag stackShotProfilerOnceFlag; \
    std::call_once(stackShotProfilerOnceFlag, [] { stackShotProfiler = new StackShotProfiler(numFrames, framesToSkip, stacksToReport); }); \
    stackShotProfiler->profile(); \
} while (false)

} // namespace WTF

WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
