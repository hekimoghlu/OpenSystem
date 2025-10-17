/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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

#include <wtf/Function.h>
#include <wtf/Noncopyable.h>
#include <wtf/OptionSet.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

#if USE(CF)
using PlatformRunLoopObserver = struct __CFRunLoopObserver*;
using PlatformRunLoop = struct __CFRunLoop*;
#else
using PlatformRunLoopObserver = void*;
using PlatformRunLoop = void*;
#endif

namespace WebCore {

class RunLoopObserver {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(RunLoopObserver, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(RunLoopObserver);
public:
    using RunLoopObserverCallback = Function<void()>;

    enum class WellKnownOrder : uint8_t {
        GraphicsCommit,
        RenderingUpdate,
        ActivityStateChange,
        InspectorFrameBegin,
        InspectorFrameEnd,
        PostRenderingUpdate,
        DisplayRefreshMonitor,
    };

    enum class Activity : uint8_t {
        BeforeWaiting   = 1 << 0,
        Entry           = 1 << 1,
        Exit            = 1 << 2,
        AfterWaiting    = 1 << 3,
    };

    enum class Type : bool { Repeating, OneShot };
    RunLoopObserver(WellKnownOrder order, RunLoopObserverCallback&& callback, Type type = Type::Repeating)
        : m_callback(WTFMove(callback))
        , m_type(type)
#if USE(CF)
        , m_order(order)
    { }
#else
    {
        UNUSED_PARAM(order);
    }
#endif

    WEBCORE_EXPORT ~RunLoopObserver();

    static constexpr OptionSet defaultActivities = { Activity::BeforeWaiting, Activity::Exit };
    WEBCORE_EXPORT void schedule(PlatformRunLoop = nullptr, OptionSet<Activity> = defaultActivities);
    WEBCORE_EXPORT void invalidate();
    WEBCORE_EXPORT bool isScheduled() const;

    bool isRepeating() const { return m_type == Type::Repeating; }

#if USE(CF)
    static void runLoopObserverFired(PlatformRunLoopObserver, unsigned long, void*);
#endif

private:
    void runLoopObserverFired();

    RunLoopObserverCallback m_callback;
    Type m_type { Type::Repeating };
#if USE(CF)
    WellKnownOrder m_order { WellKnownOrder::GraphicsCommit };
    RetainPtr<PlatformRunLoopObserver> m_runLoopObserver;
#endif
};

} // namespace WebCore
