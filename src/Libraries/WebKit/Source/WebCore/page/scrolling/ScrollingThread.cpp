/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#include "ScrollingThread.h"

#if ENABLE(SCROLLING_THREAD) || ENABLE(THREADED_ANIMATION_RESOLUTION)

#include <mutex>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>

namespace WebCore {

ScrollingThread& ScrollingThread::singleton()
{
    static LazyNeverDestroyed<ScrollingThread> scrollingThread;
    static std::once_flag onceFlag;
    std::call_once(onceFlag, [] {
        scrollingThread.construct();
    });

    return scrollingThread;
}

ScrollingThread::ScrollingThread()
    : m_runLoop(RunLoop::create("WebCore: Scrolling"_s, ThreadType::Graphics, Thread::QOS::UserInteractive))
{
}

bool ScrollingThread::isCurrentThread()
{
    return ScrollingThread::singleton().m_runLoop->isCurrent();
}

void ScrollingThread::dispatch(Function<void ()>&& function)
{
    ScrollingThread::singleton().runLoop().dispatch(WTFMove(function));
}

void ScrollingThread::dispatchBarrier(Function<void ()>&& function)
{
    dispatch([function = WTFMove(function)]() mutable {
        callOnMainThread(WTFMove(function));
    });
}

} // namespace WebCore

#endif // ENABLE(SCROLLING_THREAD) || ENABLE(THREADED_ANIMATION_RESOLUTION)
