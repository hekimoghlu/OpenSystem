/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 8, 2023.
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

#include <wtf/Condition.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {
class DisplayVBlankMonitor;
}

namespace WTF {
template<typename T> struct IsDeprecatedTimerSmartPointerException;
template<> struct IsDeprecatedTimerSmartPointerException<WebKit::DisplayVBlankMonitor> : std::true_type { };
}

namespace WebKit {

using PlatformDisplayID = uint32_t;

class DisplayVBlankMonitor {
    WTF_MAKE_TZONE_ALLOCATED(DisplayVBlankMonitor);
public:
    static std::unique_ptr<DisplayVBlankMonitor> create(PlatformDisplayID);
    virtual ~DisplayVBlankMonitor();

    enum class Type { Drm, Timer };
    virtual Type type() const = 0;

    unsigned refreshRate() const { return m_refreshRate; }

    void start();
    void stop();
    bool isActive();
    void invalidate();

    void setHandler(Function<void()>&&);

protected:
    explicit DisplayVBlankMonitor(unsigned);

    virtual bool waitForVBlank() const = 0;

    unsigned m_refreshRate;

private:
    enum class State { Stop, Active, Failed, Invalid };

    bool startThreadIfNeeded();
    void destroyThreadTimerFired();

    RefPtr<Thread> m_thread;
    Lock m_lock;
    Condition m_condition;
    State m_state WTF_GUARDED_BY_LOCK(m_lock) { State::Stop };
    Function<void()> m_handler;
    RunLoop::Timer m_destroyThreadTimer;
};

} // namespace WebKit
