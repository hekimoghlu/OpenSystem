/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
#import "config.h"
#import "RemoteLayerTreeScrollingPerformanceData.h"

#import "RemoteLayerTreeDrawingAreaProxy.h"
#import "RemoteLayerTreeHost.h"
#import "WebPageProxy.h"
#import <QuartzCore/CALayer.h>
#import <WebCore/PerformanceLoggingClient.h>
#import <WebCore/TileController.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/cocoa/VectorCocoa.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteLayerTreeScrollingPerformanceData);

RemoteLayerTreeScrollingPerformanceData::RemoteLayerTreeScrollingPerformanceData(RemoteLayerTreeDrawingAreaProxy& drawingArea)
    : m_drawingArea(drawingArea)
{
}

RemoteLayerTreeScrollingPerformanceData::~RemoteLayerTreeScrollingPerformanceData()
{
}

void RemoteLayerTreeScrollingPerformanceData::didCommitLayerTree(const FloatRect& visibleRect)
{
    // FIXME: maybe we only care about newly created tiles?
    appendBlankPixelCount(ScrollingLogEvent::Filled, blankPixelCount(visibleRect));
    logData();
}

void RemoteLayerTreeScrollingPerformanceData::didScroll(const FloatRect& visibleRect)
{
    auto pixelCount = blankPixelCount(visibleRect);
#if PLATFORM(MAC)
    if (pixelCount || m_lastUnfilledArea)
        m_lastUnfilledArea = pixelCount;
    else
        return;
#endif
    appendBlankPixelCount(ScrollingLogEvent::Exposed, pixelCount);
    logData();
}

bool RemoteLayerTreeScrollingPerformanceData::ScrollingLogEvent::canCoalesce(ScrollingLogEvent::EventType type, uint64_t pixelCount) const
{
    return eventType == type && value == pixelCount;
}

void RemoteLayerTreeScrollingPerformanceData::didChangeSynchronousScrollingReasons(WTF::MonotonicTime timestamp, uint64_t data)
{
    appendSynchronousScrollingChange(timestamp, data);
    logData();
}

void RemoteLayerTreeScrollingPerformanceData::appendBlankPixelCount(ScrollingLogEvent::EventType eventType, uint64_t blankPixelCount)
{
    auto now = MonotonicTime::now();
#if !PLATFORM(MAC)
    if (!m_events.isEmpty() && m_events.last().canCoalesce(eventType, blankPixelCount)) {
        m_events.last().endTime = now;
        return;
    }
#endif
    m_events.append(ScrollingLogEvent(now, now, eventType, blankPixelCount));
}

void RemoteLayerTreeScrollingPerformanceData::appendSynchronousScrollingChange(WTF::MonotonicTime timestamp, uint64_t scrollingChangeData)
{
    m_events.append(ScrollingLogEvent(timestamp, timestamp, ScrollingLogEvent::SwitchedScrollingMode, scrollingChangeData));
}

NSArray *RemoteLayerTreeScrollingPerformanceData::data()
{
    return createNSArray(m_events, [] (auto& pixelData) {
        return @[
            @(pixelData.startTime.toMachAbsoluteTime()),
            (pixelData.eventType == ScrollingLogEvent::Filled) ? @"filled" : @"exposed",
            @(pixelData.value)
        ];
    }).autorelease();
}

static CALayer *findTileGridContainerLayer(CALayer *layer)
{
    for (CALayer *currLayer : [layer sublayers]) {
        String layerName = [currLayer name];
        if (layerName == TileController::tileGridContainerLayerName())
            return currLayer;

        if (CALayer *foundLayer = findTileGridContainerLayer(currLayer))
            return foundLayer;
    }

    return nil;
}

unsigned RemoteLayerTreeScrollingPerformanceData::blankPixelCount(const FloatRect& visibleRect) const
{
    CALayer *rootLayer = m_drawingArea->remoteLayerTreeHost().rootLayer();

    CALayer *tileGridContainer = findTileGridContainerLayer(rootLayer);
    if (!tileGridContainer) {
        NSLog(@"Failed to find TileGrid Container Layer");
        return UINT_MAX;
    }

    FloatRect visibleRectExcludingToolbar = visibleRect;
    if (visibleRectExcludingToolbar.y() < 0)
        visibleRectExcludingToolbar.setY(0);

    Region paintedVisibleTileRegion;

    for (CALayer *tileLayer : [tileGridContainer sublayers]) {
        FloatRect tileRect = [tileLayer convertRect:[tileLayer bounds] toLayer:tileGridContainer];
    
        tileRect.intersect(visibleRectExcludingToolbar);
        
        if (!tileRect.isEmpty())
            paintedVisibleTileRegion.unite(enclosingIntRect(tileRect));
    }

    Region uncoveredRegion(enclosingIntRect(visibleRectExcludingToolbar));
    uncoveredRegion.subtract(paintedVisibleTileRegion);

    return uncoveredRegion.totalArea();
}

void RemoteLayerTreeScrollingPerformanceData::logData()
{
#if PLATFORM(MAC)
    for (auto event : m_events) {
        switch (event.eventType) {
        case ScrollingLogEvent::SwitchedScrollingMode: {
            if (RefPtr page = m_drawingArea->page())
                page->logScrollingEvent(static_cast<uint32_t>(PerformanceLoggingClient::ScrollingEvent::SwitchedScrollingMode), event.startTime, event.value);
            break;
        }
        case ScrollingLogEvent::Exposed: {
            if (RefPtr page = m_drawingArea->page())
                page->logScrollingEvent(static_cast<uint32_t>(PerformanceLoggingClient::ScrollingEvent::ExposedTilelessArea), event.startTime, event.value);
            break;
        }
        case ScrollingLogEvent::Filled: {
            if (RefPtr page = m_drawingArea->page())
                page->logScrollingEvent(static_cast<uint32_t>(PerformanceLoggingClient::ScrollingEvent::FilledTile), event.startTime, event.value);
            break;
        }
        default:
            break;
        }
    }
    m_events.clear();
#endif
}

}
