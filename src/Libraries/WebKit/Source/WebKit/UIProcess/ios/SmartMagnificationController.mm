/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#import "SmartMagnificationController.h"

#if PLATFORM(IOS_FAMILY)

#import "MessageSenderInlines.h"
#import "SmartMagnificationControllerMessages.h"
#import "ViewGestureGeometryCollectorMessages.h"
#import "WKContentView.h"
#import "WKScrollView.h"
#import "WebPageGroup.h"
#import "WebPageMessages.h"
#import "WebPageProxy.h"
#import "WebProcessProxy.h"
#import <pal/system/ios/UserInterfaceIdiom.h>
#import <wtf/TZoneMallocInlines.h>

static const float smartMagnificationPanScrollThresholdZoomedOut = 60;
static const float smartMagnificationPanScrollThresholdIPhone = 100;
static const float smartMagnificationPanScrollThresholdIPad = 150;
static const float smartMagnificationElementPadding = 0.05;

static const double smartMagnificationMaximumScale = 1.6;
static const double smartMagnificationMinimumScale = 0;

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(SmartMagnificationController);

Ref<SmartMagnificationController> SmartMagnificationController::create(WKContentView *contentView)
{
    return adoptRef(*new SmartMagnificationController(contentView));
}

SmartMagnificationController::SmartMagnificationController(WKContentView *contentView)
    : m_webPageProxy(*contentView.page)
    , m_contentView(contentView)
{
    m_webPageProxy->protectedLegacyMainFrameProcess()->addMessageReceiver(Messages::SmartMagnificationController::messageReceiverName(), m_webPageProxy->webPageIDInMainFrameProcess(), *this);
}

SmartMagnificationController::~SmartMagnificationController()
{
    if (RefPtr page = m_webPageProxy.get())
        page->protectedLegacyMainFrameProcess()->removeMessageReceiver(Messages::SmartMagnificationController::messageReceiverName(), page->webPageIDInMainFrameProcess());
}

void SmartMagnificationController::handleSmartMagnificationGesture(FloatPoint origin)
{
    if (RefPtr page = m_webPageProxy.get())
        page->protectedLegacyMainFrameProcess()->send(Messages::ViewGestureGeometryCollector::CollectGeometryForSmartMagnificationGesture(origin), page->webPageIDInMainFrameProcess());
}

void SmartMagnificationController::handleResetMagnificationGesture(FloatPoint origin)
{
    [m_contentView _zoomOutWithOrigin:origin];
}

std::tuple<FloatRect, double, double> SmartMagnificationController::smartMagnificationTargetRectAndZoomScales(FloatRect targetRect, double minimumScale, double maximumScale, bool addMagnificationPadding)
{
    FloatRect outTargetRect = targetRect;
    double outMinimumScale = minimumScale;
    double outMaximumScale = maximumScale;

    if (addMagnificationPadding) {
        outTargetRect.inflateX(smartMagnificationElementPadding * outTargetRect.width());
        outTargetRect.inflateY(smartMagnificationElementPadding * outTargetRect.height());
    }

    outMinimumScale = std::max(outMinimumScale, smartMagnificationMinimumScale);
    outMaximumScale = std::min(outMaximumScale, smartMagnificationMaximumScale);

    return { outTargetRect, outMinimumScale, outMaximumScale };
}

double SmartMagnificationController::zoomFactorForTargetRect(FloatRect targetRect, bool fitEntireRect, double viewportMinimumScale, double viewportMaximumScale)
{
    // FIXME: Share some of this code with didCollectGeometryForSmartMagnificationGesture?

    auto [adjustedTargetRect, minimumScale, maximumScale] = smartMagnificationTargetRectAndZoomScales(targetRect, viewportMinimumScale, viewportMaximumScale, !fitEntireRect);

    double currentScale = [m_contentView _contentZoomScale];
    double targetScale = [m_contentView _targetContentZoomScaleForRect:adjustedTargetRect currentScale:currentScale fitEntireRect:fitEntireRect minimumScale:minimumScale maximumScale:maximumScale];

    if (targetScale == currentScale)
        targetScale = [m_contentView _initialScaleFactor];

    return targetScale;
}

void SmartMagnificationController::didCollectGeometryForSmartMagnificationGesture(FloatPoint origin, FloatRect absoluteTargetRect, FloatRect visibleContentRect, bool fitEntireRect, double viewportMinimumScale, double viewportMaximumScale)
{
    if (absoluteTargetRect.isEmpty()) {
        // FIXME: If we don't zoom, send the tap along to text selection (see <rdar://problem/6810344>).
        [m_contentView _zoomToInitialScaleWithOrigin:origin];
        return;
    }
    RefPtr page = m_webPageProxy.get();
    if (!page)
        return;

    auto [adjustedTargetRect, minimumScale, maximumScale] = smartMagnificationTargetRectAndZoomScales(absoluteTargetRect, viewportMinimumScale, viewportMaximumScale, !fitEntireRect);

    // FIXME: Check if text selection wants to consume the double tap before we attempt magnification.

    // If the content already fits in the scroll view and we're already zoomed in to the target scale,
    // it is most likely that the user intended to scroll, so use a small distance threshold to initiate panning.
    float minimumScrollDistance;
    if ([m_contentView bounds].size.width <= page->unobscuredContentRect().width())
        minimumScrollDistance = smartMagnificationPanScrollThresholdZoomedOut;
    else if (PAL::currentUserInterfaceIdiomIsSmallScreen())
        minimumScrollDistance = smartMagnificationPanScrollThresholdIPhone;
    else
        minimumScrollDistance = smartMagnificationPanScrollThresholdIPad;

    // For replaced elements like images, we want to fit the whole element
    // in the view, so scale it down enough to make both dimensions fit if possible.
    // For other elements, try to fit them horizontally.
    if ([m_contentView _zoomToRect:adjustedTargetRect withOrigin:origin fitEntireRect:fitEntireRect minimumScale:minimumScale maximumScale:maximumScale minimumScrollDistance:minimumScrollDistance])
        return;

    // FIXME: If we still don't zoom, send the tap along to text selection (see <rdar://problem/6810344>).
    [m_contentView _zoomToInitialScaleWithOrigin:origin];
}

void SmartMagnificationController::scrollToRect(FloatPoint origin, FloatRect targetRect)
{
    [m_contentView _scrollToRect:targetRect withOrigin:origin minimumScrollDistance:0];
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
