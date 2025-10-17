/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 7, 2025.
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
#import "RemoteLayerTreeDrawingAreaProxyIOS.h"

#if PLATFORM(IOS_FAMILY)

#import "CAFrameRateRangeUtilities.h"
#import "RemoteScrollingCoordinatorProxyIOS.h"
#import "WebPageProxy.h"
#import "WebPreferences.h"
#import "WebProcessProxy.h"
#import <QuartzCore/CADisplayLink.h>
#import <WebCore/LocalFrameView.h>
#import <WebCore/ScrollView.h>
#import <pal/spi/cocoa/QuartzCoreSPI.h>
#import <wtf/TZoneMallocInlines.h>

constexpr WebCore::FramesPerSecond DisplayLinkFramesPerSecond = 60;

@interface WKDisplayLinkHandler : NSObject {
    WebKit::RemoteLayerTreeDrawingAreaProxy* _drawingAreaProxy;
    CADisplayLink *_displayLink;
#if ENABLE(TIMER_DRIVEN_DISPLAY_REFRESH_FOR_TESTING)
    RetainPtr<NSTimer> _updateTimer;
    std::optional<WebCore::FramesPerSecond> _overrideFrameRate;
#endif
}

- (id)initWithDrawingAreaProxy:(WebKit::RemoteLayerTreeDrawingAreaProxy*)drawingAreaProxy;
- (void)setPreferredFramesPerSecond:(NSInteger)preferredFramesPerSecond;
- (void)displayLinkFired:(CADisplayLink *)sender;
- (void)invalidate;
- (void)schedule;
- (WebCore::FramesPerSecond)nominalFramesPerSecond;

@end

static void* displayRefreshRateObservationContext = &displayRefreshRateObservationContext;

@implementation WKDisplayLinkHandler

- (id)initWithDrawingAreaProxy:(WebKit::RemoteLayerTreeDrawingAreaProxy*)drawingAreaProxy
{
    if (self = [super init]) {
        _drawingAreaProxy = drawingAreaProxy;
        // Note that CADisplayLink retains its target (self), so a call to -invalidate is needed on teardown.
        bool createDisplayLink = true;
#if ENABLE(TIMER_DRIVEN_DISPLAY_REFRESH_FOR_TESTING)
        NSInteger overrideRefreshRateValue = [NSUserDefaults.standardUserDefaults integerForKey:@"MainScreenRefreshRate"];
        if (overrideRefreshRateValue) {
            _overrideFrameRate = overrideRefreshRateValue;
            createDisplayLink = false;
        }
#endif
        if (createDisplayLink) {
            _displayLink = [CADisplayLink displayLinkWithTarget:self selector:@selector(displayLinkFired:)];
            [_displayLink addToRunLoop:[NSRunLoop mainRunLoop] forMode:NSRunLoopCommonModes];
            [_displayLink.display addObserver:self forKeyPath:@"refreshRate" options:NSKeyValueObservingOptionNew context:displayRefreshRateObservationContext];
            _displayLink.paused = YES;

            if (drawingAreaProxy && drawingAreaProxy->page() && !drawingAreaProxy->page()->preferences().preferPageRenderingUpdatesNear60FPSEnabled()) {
#if HAVE(CORE_ANIMATION_FRAME_RATE_RANGE)
                [_displayLink setPreferredFrameRateRange:WebKit::highFrameRateRange()];
                [_displayLink setHighFrameRateReason:WebKit::preferPageRenderingUpdatesNear60FPSDisabledHighFrameRateReason];
#else
                _displayLink.preferredFramesPerSecond = (1.0 / _displayLink.maximumRefreshRate);
#endif
            } else
                _displayLink.preferredFramesPerSecond = DisplayLinkFramesPerSecond;
        }
    }
    return self;
}

- (void)dealloc
{
    ASSERT(!_displayLink);
    [super dealloc];
}

- (void)setPreferredFramesPerSecond:(NSInteger)preferredFramesPerSecond
{
    _displayLink.preferredFramesPerSecond = preferredFramesPerSecond;
}

- (void)displayLinkFired:(CADisplayLink *)sender
{
    ASSERT(isUIThread());
    _drawingAreaProxy->didRefreshDisplay();
}

#if ENABLE(TIMER_DRIVEN_DISPLAY_REFRESH_FOR_TESTING)
- (void)timerFired
{
    ASSERT(isUIThread());
    _drawingAreaProxy->didRefreshDisplay();
}
#endif // ENABLE(TIMER_DRIVEN_DISPLAY_REFRESH_FOR_TESTING)

- (void)invalidate
{
    [_displayLink.display removeObserver:self forKeyPath:@"refreshRate" context:displayRefreshRateObservationContext];
    [_displayLink invalidate];
    _displayLink = nullptr;

#if ENABLE(TIMER_DRIVEN_DISPLAY_REFRESH_FOR_TESTING)
    [_updateTimer invalidate];
    _updateTimer = nil;
#endif
}

- (void)schedule
{
    _displayLink.paused = NO;
#if ENABLE(TIMER_DRIVEN_DISPLAY_REFRESH_FOR_TESTING)
    if (!_updateTimer && _overrideFrameRate.has_value())
        _updateTimer = [NSTimer scheduledTimerWithTimeInterval:1.0 / _overrideFrameRate.value() target:self selector:@selector(timerFired) userInfo:nil repeats:YES];
#endif
}

- (void)pause
{
    _displayLink.paused = YES;
#if ENABLE(TIMER_DRIVEN_DISPLAY_REFRESH_FOR_TESTING)
    [_updateTimer invalidate];
    _updateTimer = nil;
#endif
}

- (void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary *)change context:(void *)context
{
    if (context != displayRefreshRateObservationContext)
        return;
    [self didChangeNominalFramesPerSecond];
}

- (WebCore::FramesPerSecond)nominalFramesPerSecond
{
    RefPtr page = _drawingAreaProxy->page();
    if (page && (page->preferences().webAnimationsCustomFrameRateEnabled() || !page->preferences().preferPageRenderingUpdatesNear60FPSEnabled())) {
        auto minimumRefreshInterval = _displayLink.maximumRefreshRate;
        if (minimumRefreshInterval > 0)
            return std::round(1.0 / minimumRefreshInterval);
    }

    return DisplayLinkFramesPerSecond;
}

- (void)didChangeNominalFramesPerSecond
{
    RefPtr page = _drawingAreaProxy->page();
    if (!page)
        return;
    if (auto displayID = page->displayID())
        page->windowScreenDidChange(*displayID);
}

@end

namespace WebKit {
using namespace IPC;
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteLayerTreeDrawingAreaProxyIOS);

Ref<RemoteLayerTreeDrawingAreaProxyIOS> RemoteLayerTreeDrawingAreaProxyIOS::create(WebPageProxy& page, WebProcessProxy& webProcessProxy)
{
    return adoptRef(*new RemoteLayerTreeDrawingAreaProxyIOS(page, webProcessProxy));
}

RemoteLayerTreeDrawingAreaProxyIOS::RemoteLayerTreeDrawingAreaProxyIOS(WebPageProxy& pageProxy, WebProcessProxy& webProcessProxy)
    : RemoteLayerTreeDrawingAreaProxy(pageProxy, webProcessProxy)
{
}

RemoteLayerTreeDrawingAreaProxyIOS::~RemoteLayerTreeDrawingAreaProxyIOS()
{
    [m_displayLinkHandler invalidate];
}

std::unique_ptr<RemoteScrollingCoordinatorProxy> RemoteLayerTreeDrawingAreaProxyIOS::createScrollingCoordinatorProxy() const
{
    return makeUnique<RemoteScrollingCoordinatorProxyIOS>(*m_webPageProxy);
}

DelegatedScrollingMode RemoteLayerTreeDrawingAreaProxyIOS::delegatedScrollingMode() const
{
    return DelegatedScrollingMode::DelegatedToNativeScrollView;
}

WKDisplayLinkHandler *RemoteLayerTreeDrawingAreaProxyIOS::displayLinkHandler()
{
    if (!m_displayLinkHandler)
        m_displayLinkHandler = adoptNS([[WKDisplayLinkHandler alloc] initWithDrawingAreaProxy:this]);
    return m_displayLinkHandler.get();
}

void RemoteLayerTreeDrawingAreaProxyIOS::setPreferredFramesPerSecond(IPC::Connection& connection, FramesPerSecond preferredFramesPerSecond)
{
    if (!m_webProcessProxy->hasConnection(connection))
        return;

    [displayLinkHandler() setPreferredFramesPerSecond:preferredFramesPerSecond];
}

void RemoteLayerTreeDrawingAreaProxyIOS::didRefreshDisplay()
{
    if (m_needsDisplayRefreshCallbacksForDrawing)
        RemoteLayerTreeDrawingAreaProxy::didRefreshDisplay();

    if (m_needsDisplayRefreshCallbacksForAnimation) {
        RefPtr page = m_webPageProxy.get();
        if (!page)
            return;
        if (auto displayID = page->displayID())
            page->scrollingCoordinatorProxy()->displayDidRefresh(*displayID);
    }
}

void RemoteLayerTreeDrawingAreaProxyIOS::scheduleDisplayRefreshCallbacks()
{
    m_needsDisplayRefreshCallbacksForDrawing = true;
    [displayLinkHandler() schedule];
}

void RemoteLayerTreeDrawingAreaProxyIOS::pauseDisplayRefreshCallbacks()
{
    m_needsDisplayRefreshCallbacksForDrawing = false;
    if (!m_needsDisplayRefreshCallbacksForAnimation)
        [displayLinkHandler() pause];
}

void RemoteLayerTreeDrawingAreaProxyIOS::scheduleDisplayRefreshCallbacksForAnimation()
{
    m_needsDisplayRefreshCallbacksForAnimation = true;
    [displayLinkHandler() schedule];
}

void RemoteLayerTreeDrawingAreaProxyIOS::pauseDisplayRefreshCallbacksForAnimation()
{
    m_needsDisplayRefreshCallbacksForAnimation = false;
    if (!m_needsDisplayRefreshCallbacksForDrawing)
        [displayLinkHandler() pause];
}

std::optional<WebCore::FramesPerSecond> RemoteLayerTreeDrawingAreaProxyIOS::displayNominalFramesPerSecond()
{
    return [displayLinkHandler() nominalFramesPerSecond];
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
