/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 9, 2025.
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

#if ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)

#include "AsyncScrollingCoordinator.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ThreadedScrollingCoordinator : public AsyncScrollingCoordinator {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(ThreadedScrollingCoordinator, WEBCORE_EXPORT);
protected:
    explicit ThreadedScrollingCoordinator(Page*);
    virtual ~ThreadedScrollingCoordinator();

    void commitTreeStateIfNeeded() override;
    WEBCORE_EXPORT void hasNodeWithAnimatedScrollChanged(bool) override;
    WEBCORE_EXPORT void pageDestroyed() override;

private:
    WEBCORE_EXPORT void scheduleTreeStateCommit() final;
    WEBCORE_EXPORT void didScheduleRenderingUpdate() final;
    WEBCORE_EXPORT void willStartRenderingUpdate() final;
    WEBCORE_EXPORT void didCompleteRenderingUpdate() final;

    // Handle the wheel event on the scrolling thread. Returns whether the event was handled or not.
    WEBCORE_EXPORT WheelEventHandlingResult handleWheelEventForScrolling(const PlatformWheelEvent&, ScrollingNodeID, std::optional<WheelScrollGestureState>) final;
    WEBCORE_EXPORT void wheelEventWasProcessedByMainThread(const PlatformWheelEvent&, std::optional<WheelScrollGestureState>) final;

    WEBCORE_EXPORT void startMonitoringWheelEvents(bool clearLatchingState) final;
};

} // namespace WebCore

#endif // ENABLE(ASYNC_SCROLLING) && ENABLE(SCROLLING_THREAD)
