/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 26, 2023.
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
#import "WebChromeClient.h"

#if PLATFORM(IOS_FAMILY)

#import "DrawingArea.h"
#import "InteractionInformationAtPosition.h"
#import "InteractionInformationRequest.h"
#import "MessageSenderInlines.h"
#import "UIKitSPI.h"
#import "WebFrame.h"
#import "WebIconUtilities.h"
#import "WebPage.h"
#import "WebPageProxyMessages.h"
#import <WebCore/AudioSession.h>
#import <WebCore/ContentChangeObserver.h>
#import <WebCore/Icon.h>
#import <WebCore/MouseEvent.h>
#import <WebCore/NotImplemented.h>
#import <WebCore/PlatformMouseEvent.h>
#import <wtf/RefPtr.h>

namespace WebKit {
using namespace WebCore;

#if ENABLE(IOS_TOUCH_EVENTS)

void WebChromeClient::didPreventDefaultForEvent()
{
    RefPtr localMainFrame = page().localMainFrame();
    if (!localMainFrame)
        return;
    ContentChangeObserver::didPreventDefaultForEvent(*localMainFrame);
}

#endif

void WebChromeClient::didReceiveMobileDocType(bool isMobileDoctype)
{
    protectedPage()->didReceiveMobileDocType(isMobileDoctype);
}

void WebChromeClient::setNeedsScrollNotifications(WebCore::LocalFrame&, bool)
{
    notImplemented();
}

void WebChromeClient::didFinishContentChangeObserving(WebCore::LocalFrame&, WKContentChange observedContentChange)
{
    protectedPage()->didFinishContentChangeObserving(observedContentChange);
}

void WebChromeClient::notifyRevealedSelectionByScrollingFrame(WebCore::LocalFrame&)
{
    protectedPage()->didScrollSelection();
}

bool WebChromeClient::isStopping()
{
    notImplemented();
    return false;
}

void WebChromeClient::didLayout(LayoutType type)
{
    if (type == Scroll)
        protectedPage()->didScrollSelection();
}

void WebChromeClient::didStartOverflowScroll()
{
    // FIXME: This is only relevant for legacy touch-driven overflow in the web process (see ScrollAnimatorIOS::handleTouchEvent), and should be removed.
    protectedPage()->send(Messages::WebPageProxy::ScrollingNodeScrollWillStartScroll(std::nullopt));
}

void WebChromeClient::didEndOverflowScroll()
{
    // FIXME: This is only relevant for legacy touch-driven overflow in the web process (see ScrollAnimatorIOS::handleTouchEvent), and should be removed.
    protectedPage()->send(Messages::WebPageProxy::ScrollingNodeScrollDidEndScroll(std::nullopt));
}

bool WebChromeClient::hasStablePageScaleFactor() const
{
    return protectedPage()->hasStablePageScaleFactor();
}

void WebChromeClient::suppressFormNotifications()
{
    notImplemented();
}

void WebChromeClient::restoreFormNotifications()
{
    notImplemented();
}

void WebChromeClient::addOrUpdateScrollingLayer(WebCore::Node*, PlatformLayer*, PlatformLayer*, const WebCore::IntSize&, bool, bool)
{
    notImplemented();
}

void WebChromeClient::removeScrollingLayer(WebCore::Node*, PlatformLayer*, PlatformLayer*)
{
    notImplemented();
}

void WebChromeClient::webAppOrientationsUpdated()
{
    notImplemented();
}

void WebChromeClient::showPlaybackTargetPicker(bool hasVideo, WebCore::RouteSharingPolicy policy, const String& routingContextUID)
{
    auto page = protectedPage();
    page->send(Messages::WebPageProxy::ShowPlaybackTargetPicker(hasVideo, page->rectForElementAtInteractionLocation(), policy, routingContextUID));
}

Seconds WebChromeClient::eventThrottlingDelay()
{
    return protectedPage()->eventThrottlingDelay();
}

#if ENABLE(ORIENTATION_EVENTS)
IntDegrees WebChromeClient::deviceOrientation() const
{
    return protectedPage()->deviceOrientation();
}
#endif

bool WebChromeClient::shouldUseMouseEventForSelection(const WebCore::PlatformMouseEvent& event)
{
    // In iPadOS and macCatalyst, despite getting mouse events, we still want UITextInteraction and friends to own selection gestures.
    // However, we need to allow single-clicks to set the selection, because that is how UITextInteraction is activated.
#if HAVE(UIKIT_WITH_MOUSE_SUPPORT)
    return event.clickCount() <= 1;
#else
    return true;
#endif
}

bool WebChromeClient::showDataDetectorsUIForElement(const Element& element, const Event& event)
{
    auto* mouseEvent = dynamicDowncast<MouseEvent>(event);
    if (!mouseEvent)
        return false;

    // FIXME: Ideally, we would be able to generate InteractionInformationAtPosition without re-hit-testing the element.
    auto request = InteractionInformationRequest { roundedIntPoint(mouseEvent->locationInRootViewCoordinates()) };
    request.includeLinkIndicator = true;
    auto page = protectedPage();
    auto positionInformation = page->positionInformation(request);
    page->send(Messages::WebPageProxy::ShowDataDetectorsUIForPositionInformation(positionInformation));
    return true;
}

void WebChromeClient::relayAccessibilityNotification(const String& notificationName, const RetainPtr<NSData>& notificationData) const
{
    return protectedPage()->relayAccessibilityNotification(notificationName, notificationData);
}

} // namespace WebKit

#endif // PLATFORM(IOS_FAMILY)
