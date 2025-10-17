/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 13, 2024.
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

#if ENABLE(SCROLLING_THREAD) || ENABLE(THREADED_ANIMATION_RESOLUTION)

#include <functional>
#include <wtf/Condition.h>
#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/RunLoop.h>
#include <wtf/Threading.h>
#include <wtf/Vector.h>

namespace WebCore {

class ScrollingThread {
    WTF_MAKE_NONCOPYABLE(ScrollingThread);

public:
    WEBCORE_EXPORT static bool isCurrentThread();
    WEBCORE_EXPORT static void dispatch(Function<void ()>&&);

    // Will dispatch the given function on the main thread once all pending functions
    // on the scrolling thread have finished executing. Used for synchronization purposes.
    WEBCORE_EXPORT static void dispatchBarrier(Function<void ()>&&);

private:
    friend LazyNeverDestroyed<ScrollingThread>;

    static ScrollingThread& singleton();

    ScrollingThread();

    RunLoop& runLoop() { return m_runLoop; }

    Ref<RunLoop> m_runLoop;
};

} // namespace WebCore

#endif // ENABLE(SCROLLING_THREAD) || ENABLE(THREADED_ANIMATION_RESOLUTION)
