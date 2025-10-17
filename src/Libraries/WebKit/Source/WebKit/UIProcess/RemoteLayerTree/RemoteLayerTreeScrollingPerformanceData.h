/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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

#import <WebCore/FloatRect.h>
#import <wtf/MonotonicTime.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/Vector.h>

namespace WebKit {

class RemoteLayerTreeDrawingAreaProxy;

class RemoteLayerTreeScrollingPerformanceData {
    WTF_MAKE_TZONE_ALLOCATED(RemoteLayerTreeScrollingPerformanceData);
public:
    RemoteLayerTreeScrollingPerformanceData(RemoteLayerTreeDrawingAreaProxy&);
    ~RemoteLayerTreeScrollingPerformanceData();

    void didCommitLayerTree(const WebCore::FloatRect& visibleRect);
    void didScroll(const WebCore::FloatRect& visibleRect);
    void didChangeSynchronousScrollingReasons(WTF::MonotonicTime, uint64_t scrollingChangeData);

    NSArray *data(); // Array of [ time, event type, unfilled pixel count ]
    void logData();

private:
    struct ScrollingLogEvent {
        enum EventType { Filled, Exposed, SwitchedScrollingMode };

        WTF::MonotonicTime startTime;
        WTF::MonotonicTime endTime;
        EventType eventType;
        uint64_t value;
        
        ScrollingLogEvent(WTF::MonotonicTime start, WTF::MonotonicTime end, EventType type, uint64_t data)
            : startTime(start)
            , endTime(end)
            , eventType(type)
            , value(data)
        { }
        
        bool canCoalesce(ScrollingLogEvent::EventType, uint64_t blankPixelCount) const;
    };
    
    unsigned blankPixelCount(const WebCore::FloatRect& visibleRect) const;

    void appendBlankPixelCount(ScrollingLogEvent::EventType, uint64_t blankPixelCount);
    void appendSynchronousScrollingChange(WTF::MonotonicTime, uint64_t);

    CheckedRef<RemoteLayerTreeDrawingAreaProxy> m_drawingArea;
    Vector<ScrollingLogEvent> m_events;
#if PLATFORM(MAC)
    uint64_t m_lastUnfilledArea;
#endif
};

}
