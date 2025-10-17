/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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

#include <wtf/Deque.h>
#include <wtf/Function.h>
#include <wtf/MonotonicTime.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class FrameRateMonitor {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(FrameRateMonitor, WEBCORE_EXPORT);
public:
    struct LateFrameInfo {
        MonotonicTime frameTime;
        MonotonicTime lastFrameTime;
        double observedFrameRate { 0 };
        size_t frameCount { 0 };
    };
    using LateFrameCallback = Function<void(LateFrameInfo)>;
    explicit FrameRateMonitor(LateFrameCallback&&);

    WEBCORE_EXPORT void update();

    double observedFrameRate() const { return m_observedFrameRate; }
    size_t frameCount() const { return m_frameCount; }

private:
    LateFrameCallback m_lateFrameCallback;
    Deque<double, 120> m_observedFrameTimeStamps;
    double m_observedFrameRate { 0 };
    size_t m_frameCount { 0 };
};

inline FrameRateMonitor::FrameRateMonitor(LateFrameCallback&& callback)
    : m_lateFrameCallback(WTFMove(callback))
{
}

}
