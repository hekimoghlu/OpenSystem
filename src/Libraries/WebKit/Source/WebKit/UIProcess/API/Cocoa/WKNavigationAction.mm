/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#import "WKNavigationActionInternal.h"

#import "APIHitTestResult.h"
#import "NavigationActionData.h"
#import "WKFrameInfoInternal.h"
#import "WKNavigationInternal.h"
#import "WebEventFactory.h"
#import "WebsiteDataStore.h"
#import "_WKHitTestResultInternal.h"
#import "_WKUserInitiatedActionInternal.h"
#import <WebCore/FloatPoint.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/RetainPtr.h>

#if PLATFORM(IOS_FAMILY)
#import "WebIOSEventFactory.h"
#endif

@implementation WKNavigationAction

static WKNavigationType toWKNavigationType(WebCore::NavigationType navigationType)
{
    switch (navigationType) {
    case WebCore::NavigationType::LinkClicked:
        return WKNavigationTypeLinkActivated;
    case WebCore::NavigationType::FormSubmitted:
        return WKNavigationTypeFormSubmitted;
    case WebCore::NavigationType::BackForward:
        return WKNavigationTypeBackForward;
    case WebCore::NavigationType::Reload:
        return WKNavigationTypeReload;
    case WebCore::NavigationType::FormResubmitted:
        return WKNavigationTypeFormResubmitted;
    case WebCore::NavigationType::Other:
        return WKNavigationTypeOther;
    }

    ASSERT_NOT_REACHED();
    return WKNavigationTypeOther;
}

#if PLATFORM(IOS_FAMILY)
static WKSyntheticClickType toWKSyntheticClickType(WebKit::WebMouseEventSyntheticClickType syntheticClickType)
{
    switch (syntheticClickType) {
    case WebKit::WebMouseEventSyntheticClickType::NoTap:
        return WKSyntheticClickTypeNoTap;
    case WebKit::WebMouseEventSyntheticClickType::OneFingerTap:
        return WKSyntheticClickTypeOneFingerTap;
    case WebKit::WebMouseEventSyntheticClickType::TwoFingerTap:
        return WKSyntheticClickTypeTwoFingerTap;
    }
    ASSERT_NOT_REACHED();
    return WKSyntheticClickTypeNoTap;
}
#endif

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(WKNavigationAction.class, self))
        return;

    _navigationAction->~NavigationAction();

    [super dealloc];
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@: %p; navigationType = %ld; syntheticClickType = %ld; position x = %.2f y = %.2f request = %@; sourceFrame = %@; targetFrame = %@>", NSStringFromClass(self.class), self,
        (long)self.navigationType,
#if PLATFORM(IOS_FAMILY)
        (long)self._syntheticClickType, self._clickLocationInRootViewCoordinates.x, self._clickLocationInRootViewCoordinates.y,
#else
        0L, 0.0, 0.0,
#endif
        self.request, self.sourceFrame, self.targetFrame];
}

- (WKFrameInfo *)sourceFrame
{
    return wrapper(_navigationAction->sourceFrame());
}

- (WKFrameInfo *)targetFrame
{
    return wrapper(_navigationAction->targetFrame());
}

- (WKNavigationType)navigationType
{
    return toWKNavigationType(_navigationAction->navigationType());
}

- (NSURLRequest *)request
{
    return _navigationAction->request().nsURLRequest(WebCore::HTTPBodyUpdatePolicy::UpdateHTTPBody);
}

- (BOOL)shouldPerformDownload
{
    return _navigationAction->shouldPerformDownload();
}

#if PLATFORM(IOS_FAMILY)
- (WKSyntheticClickType)_syntheticClickType
{
    return toWKSyntheticClickType(_navigationAction->syntheticClickType());
}

- (CGPoint)_clickLocationInRootViewCoordinates
{
    return _navigationAction->clickLocationInRootViewCoordinates();
}
#endif

#if PLATFORM(MAC)

- (NSEventModifierFlags)modifierFlags
{
    return WebKit::WebEventFactory::toNSEventModifierFlags(_navigationAction->modifiers());
}

- (NSInteger)buttonNumber
{
    return WebKit::WebEventFactory::toNSButtonNumber(_navigationAction->mouseButton());
}

#else

- (UIKeyModifierFlags)modifierFlags
{
    return WebKit::WebIOSEventFactory::toUIKeyModifierFlags(_navigationAction->modifiers());
}

- (UIEventButtonMask)buttonNumber
{
    return WebKit::WebIOSEventFactory::toUIEventButtonMask(_navigationAction->mouseButton());
}

#endif

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_navigationAction;
}

@end

@implementation WKNavigationAction (WKPrivate)

- (NSURL *)_originalURL
{
    return _navigationAction->originalURL();
}

- (BOOL)_isUserInitiated
{
    return _navigationAction->isProcessingUserGesture();
}

- (BOOL)_canHandleRequest
{
    return _navigationAction->canHandleRequest();
}

- (BOOL)_shouldOpenExternalSchemes
{
    return _navigationAction->shouldOpenExternalSchemes();
}

- (BOOL)_shouldOpenAppLinks
{
    return _navigationAction->shouldOpenAppLinks();
}

- (BOOL)_shouldPerformDownload
{
    return _navigationAction->shouldPerformDownload();
}

- (BOOL)_shouldOpenExternalURLs
{
    return [self _shouldOpenExternalSchemes];
}

- (_WKUserInitiatedAction *)_userInitiatedAction
{
    return wrapper(_navigationAction->userInitiatedAction());
}

- (BOOL)_isRedirect
{
    return _navigationAction->isRedirect();
}

- (WKNavigation *)_mainFrameNavigation
{
    return wrapper(_navigationAction->mainFrameNavigation());
}


- (void)_storeSKAdNetworkAttribution
{
    auto* mainFrameNavigation = _navigationAction->mainFrameNavigation();
    if (!mainFrameNavigation)
        return;
    auto& privateClickMeasurement = mainFrameNavigation->privateClickMeasurement();
    if (!privateClickMeasurement || !privateClickMeasurement->isSKAdNetworkAttribution())
        return;
    auto* sourceFrame = _navigationAction->sourceFrame();
    if (!sourceFrame)
        return;
    auto* page = sourceFrame->page();
    if (!page)
        return;
    page->websiteDataStore().storePrivateClickMeasurement(*privateClickMeasurement);
}

- (_WKHitTestResult *)_hitTestResult
{
#if PLATFORM(MAC) || HAVE(UIKIT_WITH_MOUSE_SUPPORT)
    auto& webHitTestResultData = _navigationAction->webHitTestResultData();
    if (!webHitTestResultData)
        return nil;
    RefPtr sourceFrame = _navigationAction->sourceFrame();
    if (!sourceFrame)
        return nil;
    RefPtr page = sourceFrame->page();
    if (!page)
        return nil;

    auto apiHitTestResult = API::HitTestResult::create(webHitTestResultData.value(), page.get());
    return retainPtr(wrapper(apiHitTestResult)).autorelease();
#else
    return nil;
#endif
}

- (NSString *)_targetFrameName
{
    auto& name = _navigationAction->targetFrameName();
    if (name.isNull())
        return nil;
    return name;
}

- (BOOL)_hasOpener
{
    return _navigationAction->hasOpener();
}

@end
