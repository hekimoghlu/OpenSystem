/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

#if ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)

#include "GStreamerCommon.h"
#include <functional>
#include <wtf/Deque.h>

namespace WebCore {

class TrackQueue {
public:
    TrackQueue(TrackID);

    typedef std::function<void(GRefPtr<GstMiniObject>&&)> NotEmptyHandler;
    typedef std::function<void()> LowLevelHandler;

    // Note: The TrackQueue methods are not thread-safe. TrackQueue must always be wrapped in a DataMutex<>.

    // For producer thread (main-thread):
    void enqueueObject(GRefPtr<GstMiniObject>&&);
    bool isFull() const { return durationEnqueued() >= s_durationEnqueuedHighWaterLevel; }
    void notifyWhenLowLevel(LowLevelHandler&&);
    void clear();
    void flush();

    // For consumer thread:
    bool isEmpty() const { return m_queue.isEmpty(); }
    GRefPtr<GstMiniObject> pop();
    void notifyWhenNotEmpty(NotEmptyHandler&&);
    bool hasNotEmptyHandler() const { return m_notEmptyCallback != nullptr; }
    void resetNotEmptyHandler();

private:
    // The point of having a queue for WebKitMediaSource is to limit the number of context switches per second.
    // If we had no queue, the main thread would have to be awaken for every frame. On the other hand, if the
    // queue had unlimited size WebKit would end up requesting flushes more often than necessary when frames
    // in the future are re-appended. As a sweet spot between these extremes we choose to allow enqueueing a
    // few seconds worth of samples.

    // `isReadyForMoreSamples` follows the classical two water levels strategy: initially it's true until the
    // high water level is reached, then it becomes false until the queue drains down to the low water level
    // and the cycle repeats. This way we avoid stalls and minimize context switches.

    static const GstClockTime s_durationEnqueuedHighWaterLevel = 5 * GST_SECOND;
    static const GstClockTime s_durationEnqueuedLowWaterLevel = 2 * GST_SECOND;

    GstClockTime durationEnqueued() const;
    void checkLowLevel();

    TrackID m_trackId;
    Deque<GRefPtr<GstMiniObject>> m_queue;
    LowLevelHandler m_lowLevelCallback;
    NotEmptyHandler m_notEmptyCallback;
};

}

#endif
