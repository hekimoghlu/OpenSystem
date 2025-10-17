/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#import "SOAuthorizationCoordinator.h"

#if HAVE(APP_SSO)

#import "APIFrameHandle.h"
#import "APINavigationAction.h"
#import "PopUpSOAuthorizationSession.h"
#import "RedirectSOAuthorizationSession.h"
#import "SubFrameSOAuthorizationSession.h"
#import "WKSOAuthorizationDelegate.h"
#import "WebFrameProxy.h"
#import "WebPageProxy.h"
#import <WebCore/ResourceRequest.h>
#import <pal/spi/cf/CFNetworkSPI.h>
#import <pal/spi/cocoa/AuthKitSPI.h>
#import <wtf/Function.h>
#import <wtf/TZoneMallocInlines.h>
#import <pal/cocoa/AppSSOSoftLink.h>

#define AUTHORIZATIONCOORDINATOR_RELEASE_LOG(fmt, ...) RELEASE_LOG(AppSSO, "%p - SOAuthorizationCoordinator::" fmt, this, ##__VA_ARGS__)
#define AUTHORIZATIONCOORDINATOR_RELEASE_LOG_ERROR(fmt, ...) RELEASE_LOG_ERROR(AppSSO, "%p - SOAuthorizationCoordinator::" fmt, this, ##__VA_ARGS__)

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SOAuthorizationCoordinator);

SOAuthorizationCoordinator::SOAuthorizationCoordinator()
{
    m_hasAppSSO = !!PAL::getSOAuthorizationClass();
#if PLATFORM(MAC)
    // In the case of base system, which doesn't have AppSSO.framework.
    if (!m_hasAppSSO)
        return;
#endif
    m_soAuthorizationDelegate = adoptNS([[WKSOAuthorizationDelegate alloc] init]);
    [NSURLSession _disableAppSSO];
}

bool SOAuthorizationCoordinator::canAuthorize(const URL& url) const
{
    return m_hasAppSSO && [PAL::getSOAuthorizationClass() canPerformAuthorizationWithURL:url responseCode:0];
}

void SOAuthorizationCoordinator::tryAuthorize(Ref<API::NavigationAction>&& navigationAction, WebPageProxy& page, Function<void(bool)>&& completionHandler)
{
    AUTHORIZATIONCOORDINATOR_RELEASE_LOG("tryAuthorize");
    if (!canAuthorize(navigationAction->request().url())) {
        AUTHORIZATIONCOORDINATOR_RELEASE_LOG("tryAuthorize: The requested URL is not registered for AppSSO handling. No further action needed.");
        completionHandler(false);
        return;
    }

    // SubFrameSOAuthorizationSession should only be allowed for Apple first parties.
    RefPtr targetFrame = navigationAction->targetFrame();
    bool subframeNavigation = targetFrame && !targetFrame->isMainFrame();
    if (subframeNavigation && (!page.mainFrame() || ![AKAuthorizationController isURLFromAppleOwnedDomain:page.mainFrame()->url()])) {
        AUTHORIZATIONCOORDINATOR_RELEASE_LOG_ERROR("tryAuthorize: Attempting to perform subframe navigation for non-Apple authorization URL.");
        completionHandler(false);
        return;
    }

    auto session = subframeNavigation ? SubFrameSOAuthorizationSession::create(m_soAuthorizationDelegate, WTFMove(navigationAction), page, WTFMove(completionHandler), targetFrame->handle()->frameID()) : RedirectSOAuthorizationSession::create(m_soAuthorizationDelegate, WTFMove(navigationAction), page, WTFMove(completionHandler));
    [m_soAuthorizationDelegate setSession:WTFMove(session)];
}

void SOAuthorizationCoordinator::tryAuthorize(Ref<API::PageConfiguration>&& configuration, Ref<API::NavigationAction>&& navigationAction, WebPageProxy& page, NewPageCallback&& newPageCallback, UIClientCallback&& uiClientCallback)
{
    AUTHORIZATIONCOORDINATOR_RELEASE_LOG("tryAuthorize (2)");
    if (!canAuthorize(navigationAction->request().url())) {
        AUTHORIZATIONCOORDINATOR_RELEASE_LOG("tryAuthorize (2): The requested URL is not registered for AppSSO handling. No further action needed.");
        uiClientCallback(WTFMove(navigationAction), WTFMove(newPageCallback));
        return;
    }

    bool subframeNavigation = navigationAction->sourceFrame() && !navigationAction->sourceFrame()->isMainFrame();
    if (subframeNavigation) {
        AUTHORIZATIONCOORDINATOR_RELEASE_LOG_ERROR("tryAuthorize (2): Attempting to perform subframe navigation.");
        uiClientCallback(WTFMove(navigationAction), WTFMove(newPageCallback));
        return;
    }

    if (!navigationAction->isProcessingUserGesture()) {
        AUTHORIZATIONCOORDINATOR_RELEASE_LOG_ERROR("tryAuthorize (2): Attempting to perform auth without a user gesture.");
        uiClientCallback(WTFMove(navigationAction), WTFMove(newPageCallback));
        return;
    }

    auto session = PopUpSOAuthorizationSession::create(WTFMove(configuration), m_soAuthorizationDelegate, page, WTFMove(navigationAction), WTFMove(newPageCallback), WTFMove(uiClientCallback));
    [m_soAuthorizationDelegate setSession:WTFMove(session)];
}

} // namespace WebKit

#undef AUTHORIZATIONCOORDINATOR_RELEASE_LOG_ERROR
#undef AUTHORIZATIONCOORDINATOR_RELEASE_LOG

#endif
