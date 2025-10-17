/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#import "WebFrameNetworkingContext.h"

#import "NetworkStorageSessionMap.h"
#import "WebFrameInternal.h"
#import "WebResourceLoadScheduler.h"
#import "WebViewPrivate.h"
#import <WebCore/FrameLoader.h>
#import <WebCore/LocalFrameLoaderClient.h>
#import <WebCore/NetworkStorageSession.h>
#import <WebCore/Page.h>
#import <WebCore/ResourceError.h>
#import <WebCore/Settings.h>
#import <pal/SessionID.h>
#import <pal/spi/cf/CFNetworkSPI.h>

#if PLATFORM(IOS_FAMILY)
#import <WebCore/WebCoreThread.h>
#import <WebKitLegacy/WebFrameLoadDelegate.h>
#endif

using namespace WebCore;

NetworkStorageSession& WebFrameNetworkingContext::ensurePrivateBrowsingSession()
{
    ASSERT(isMainThread());
    NetworkStorageSessionMap::ensureSession(PAL::SessionID::legacyPrivateSessionID(), [[NSBundle mainBundle] bundleIdentifier]);
    return *NetworkStorageSessionMap::storageSession(PAL::SessionID::legacyPrivateSessionID());
}

void WebFrameNetworkingContext::destroyPrivateBrowsingSession()
{
    ASSERT(isMainThread());
    NetworkStorageSessionMap::destroySession(PAL::SessionID::legacyPrivateSessionID());
}

bool WebFrameNetworkingContext::localFileContentSniffingEnabled() const
{
    return frame() && frame()->settings().localFileContentSniffingEnabled();
}

SchedulePairHashSet* WebFrameNetworkingContext::scheduledRunLoopPairs() const
{
    if (!frame() || !frame()->page())
        return nullptr;
    return frame()->page()->scheduledRunLoopPairs();
}

RetainPtr<CFDataRef> WebFrameNetworkingContext::sourceApplicationAuditData() const
{
    if (!frame() || !frame()->page())
        return nullptr;
    
    WebView *webview = kit(frame()->page());
    if (!webview)
        return nullptr;

    return (__bridge CFDataRef)webview._sourceApplicationAuditData;
}

String WebFrameNetworkingContext::sourceApplicationIdentifier() const
{
    return emptyString();
}

ResourceError WebFrameNetworkingContext::blockedError(const ResourceRequest& request) const
{
    return WebResourceLoadScheduler::blockedErrorFromRequest(request);
}

NetworkStorageSession* WebFrameNetworkingContext::storageSession() const
{
    ASSERT(isMainThread());
    if (frame() && frame()->page() && frame()->page()->sessionID().isEphemeral()) {
        if (auto* session = NetworkStorageSessionMap::storageSession(PAL::SessionID::legacyPrivateSessionID()))
            return session;
        // Some requests may still be coming shortly before WebCore updates the session ID and after WebKit destroys the private browsing session.
        LOG_ERROR("Invalid session ID. Please file a bug unless you just disabled private browsing, in which case it's an expected race.");
    }
    return &NetworkStorageSessionMap::defaultStorageSession();
}
