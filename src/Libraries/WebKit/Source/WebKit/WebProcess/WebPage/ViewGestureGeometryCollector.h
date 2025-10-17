/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 27, 2024.
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
#ifndef ViewGestureGeometryCollector_h
#define ViewGestureGeometryCollector_h

#include "MessageReceiver.h"
#include <WebCore/PageIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {
class FloatPoint;
class FloatRect;
class Node;
}

namespace WebKit {

class WebPage;

class ViewGestureGeometryCollector : private IPC::MessageReceiver, public RefCounted<ViewGestureGeometryCollector> {
    WTF_MAKE_TZONE_ALLOCATED(ViewGestureGeometryCollector);
public:
    static Ref<ViewGestureGeometryCollector> create(WebPage& webPage)
    {
        return adoptRef(*new ViewGestureGeometryCollector(webPage));
    }

    ~ViewGestureGeometryCollector();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void mainFrameDidLayout();

    void computeZoomInformationForNode(WebCore::Node&, WebCore::FloatPoint& origin, WebCore::FloatRect& absoluteBoundingRect, bool& isReplaced, double& viewportMinimumScale, double& viewportMaximumScale);

private:
    explicit ViewGestureGeometryCollector(WebPage&);

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Message handlers.
    void collectGeometryForSmartMagnificationGesture(WebCore::FloatPoint gestureLocationInViewCoordinates);

#if !PLATFORM(IOS_FAMILY)
    void collectGeometryForMagnificationGesture();

    void setRenderTreeSizeNotificationThreshold(uint64_t);
    void sendDidHitRenderTreeSizeThresholdIfNeeded();
#endif

    void dispatchDidCollectGeometryForSmartMagnificationGesture(WebCore::FloatPoint origin, WebCore::FloatRect absoluteTargetRect, WebCore::FloatRect visibleContentRect, bool fitEntireRect, double viewportMinimumScale, double viewportMaximumScale);
    void computeMinimumAndMaximumViewportScales(double& viewportMinimumScale, double& viewportMaximumScale) const;

#if PLATFORM(IOS_FAMILY)
    std::optional<std::pair<double, double>> computeTextLegibilityScales(double& viewportMinimumScale, double& viewportMaximumScale);
#endif

    WeakPtr<WebPage> m_webPage;
    WebCore::PageIdentifier m_webPageIdentifier;

#if !PLATFORM(IOS_FAMILY)
    uint64_t m_renderTreeSizeNotificationThreshold;
#else
    std::optional<std::pair<double, double>> m_cachedTextLegibilityScales;
#endif
};

} // namespace WebKit

#endif // ViewGestureGeometryCollector
